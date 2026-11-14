"""Track both wrist endpoints; preserve raw images, valid poses and explicit gaps."""
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
import shutil
import threading
import time
import numpy as np

from ..common.paths import OUTPUT
from ..common.runtime import FrameClock, LatestPairs
from ..common.sources import Source
from .detector import Detector
from .tracking import Track
from .pose import rotation_distance
from .bundle import Calibrator, marker_ids, save_calibrations
from .diagnostics import ObservationDiagnostics


def run(cfg, cameras, args):
    session = Path(args.session).expanduser().resolve() if args.session else OUTPUT/"hand_pose"/datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    session.mkdir(parents=True, exist_ok=False)
    calibration = Path(cfg["calibration"]["file"])
    shutil.copy2(calibration, session/"calibration.yaml")
    metadata = {
        "version": 2, "mode": "synthetic_demo" if args.demo else args.command,
        "head_frame": cfg["reference"], "head_axes": "X right, Y down, Z forward; head-mounted reference camera",
        "pose": "T_head_endpoint maps endpoint coordinates into the CURRENT head camera frame, metres",
        "trajectory": "head-relative; head motion is not removed; no VIO/world frame",
        "marker_axes": "decoded tag X right, Y up, Z outward from printed face",
        "wrist_axes": "origin: estimated marker-loop centre; +Z: approximate palm seam; +X: last-to-first ID across seam; +Y: Z cross X; not anatomical ground truth",
        "quaternion": "xyzw", "config": cfg,
        "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
        "clock": "host_receive" if cfg["input"]["type"] == "cyper" and not args.demo else "media_time",
        "finished": False,
    }
    (session/"session.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+"\n")
    print(f"[session] {session}", flush=True)
    print(f"[pose] head={cfg['reference']}; dictionary={cfg['dictionary']}; black border={cfg['border_bits']} cells", flush=True)
    print("[pose] 末端坐标相对于头部；多码模式原点是估计的腕部中心，接缝近似定手心方向。丢码不外推。", flush=True)
    if args.demo:
        print("[demo] 合成鱼眼图像 → 实际识码 → 实际位姿解算；不是实机测量。", flush=True)
    detector = Detector(cfg, cameras)
    tracks = {t["name"]: Track(t, cfg["tracking"]) for t in cfg["targets"]}
    calibrators = ({t["name"]: Calibrator(t, cfg["bundle_calibration_quality"]) for t in cfg["targets"]}
                   if args.command == "calibrate" else {})
    diagnostics = ObservationDiagnostics(cfg)
    inspection = bool(calibrators) or args.command == "check"
    best_score = -1
    if calibrators:
        print("[calibrate] 缓慢转腕，让每对相邻标签（包括首尾接缝）同时被两眼看到；保持腕带形状不变。", flush=True)
    clock, stop, capture = FrameClock(), threading.Event(), None
    frames, valid_counts, errors = 0, {name: 0 for name in tracks}, []
    with ExitStack() as resources:
        viewer = None
        if not args.no_viz:
            from .visualizer import Visualizer
            viewer = Visualizer(cfg, args, session/"viewer.rrd")
            resources.callback(viewer.close)
        table = resources.enter_context((session/"poses.csv").open("w", newline=""))
        writer = csv.writer(table)
        columns = ["time_s", "source_time_s", "frame", "target", "marker_id", "valid", "status", "segment",
                   "reprojection_px", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw",
                   "raw_x_m", "raw_y_m", "raw_z_m", "raw_qx", "raw_qy", "raw_qz", "raw_qw", "used_marker_ids"]
        writer.writerow(columns)
        sample_file = resources.enter_context((session/"marker_samples.jsonl").open("w")) if inspection else None
        truth_file = resources.enter_context((session/"ground_truth.jsonl").open("w")) if args.demo else None
        if args.demo:
            from .demo import DemoSource
            source = DemoSource(cfg, cameras)
        else:
            source = Source(cfg, cameras)
        if source.kind in ("cyper", "uvc", "ros2"):
            capture = LatestPairs(source, stop)
            resources.callback(capture.close)
            capture.start()
            read = capture.read
        else:
            resources.callback(source.close)
            source.open()
            read = lambda: source.read(stop)
        started = time.monotonic()
        last_report = started - 2
        try:
            while not stop.is_set():
                if args.frames and frames >= args.frames:
                    break
                if args.seconds and time.monotonic() - started >= args.seconds:
                    break
                pair = read()
                if pair is None:
                    break
                stamp, _ = clock.update(pair.stamp)
                if args.command == "scan":
                    detector.observe(pair.images)
                    results = {name: (None, "scan_only") for name in tracks}
                elif inspection:
                    measurements = detector.measure(pair.images)
                    for calibrator in calibrators.values():
                        calibrator.add(stamp, measurements)
                    diagnostics.update(stamp, detector.last_seen, measurements)
                    sample_file.write(json.dumps({"time_s": stamp, "decoded": detector.last_seen,
                        "rejected": {str(i): reason for i, (m, reason) in measurements.items() if m is None and reason != "not_seen"},
                        "markers": {
                        str(i): {"T_head_marker": m.T_head_marker.tolist(), "reprojection_px": m.reprojection_px,
                                 "mode": m.mode} for i, (m, _) in measurements.items() if m is not None}})+"\n")
                    results = {name: (None, "calibrating" if calibrators else "checking") for name in tracks}
                    configured_ids = {i for target in cfg["targets"] for i in marker_ids(target)}
                    target_seen = {int(item["id"]) for item in detector.last_seen} & configured_ids
                    score = len(diagnostics.last_valid)*100 + len(target_seen)
                    if score > best_score:
                        import cv2
                        for channel, image in zip(cfg["channels"], pair.images):
                            cv2.imwrite(str(session/f"check-{channel['camera']}.jpg"), image)
                        (session/"check-frame.json").write_text(json.dumps({"time_s": stamp, "decoded": detector.last_seen}, indent=2)+"\n")
                        best_score = score
                else:
                    results = detector.process(pair.images)
                for name, track in tracks.items():
                    measurement, reason = results[name]
                    track.update(stamp, measurement, reason)
                    row = [stamp, pair.stamp, pair.sequence, name, ";".join(map(str, marker_ids(track.target))), int(track.pose is not None),
                           track.status, track.segment]
                    if track.pose is None:
                        row += [""] * 15
                    else:
                        valid_counts[name] += 1
                        row += [measurement.reprojection_px, *track.pose[:3, 3], *track.quaternion,
                                *track.raw_pose[:3, 3], *track.raw_quaternion]
                        if args.demo:
                            truth = source.truth[name] @ track.offset
                            errors.append({"target": name, "position_m": float(np.linalg.norm(track.raw_pose[:3, 3]-truth[:3, 3])),
                                           "rotation_deg": float(np.rad2deg(rotation_distance(track.raw_pose[:3, :3], truth[:3, :3])))})
                    row += [";".join(map(str, track.measurement.marker_ids)) if track.measurement else ""]
                    writer.writerow(row)
                if truth_file:
                    truth_file.write(json.dumps({"time_s": stamp, "T_head_target": {name: p.tolist() for name, p in source.truth.items()}})+"\n")
                if viewer:
                    viewer.log(pair, stamp, tracks)
                frames += 1
                now = time.monotonic()
                if now-last_report >= 1:
                    if args.command == "scan":
                        print("[scan] "+json.dumps(detector.last_seen, ensure_ascii=False), flush=True)
                    elif inspection:
                        print("[check] "+diagnostics.line(), flush=True)
                        if calibrators:
                            print("[pairs] "+json.dumps({name: c.progress() for name, c in calibrators.items()}), flush=True)
                    else:
                        states = " | ".join(f"{name}: {track.status}" for name, track in tracks.items())
                        print(f"[pose] frames={frames} | {states}", flush=True)
                    table.flush()
                    last_report = now
                if calibrators and diagnostics.stalled():
                    print("[calibrate] 连续 15 秒没有有效位姿，提前停止。先运行 check 检查识码和双目有效性。", flush=True)
                    break
        except KeyboardInterrupt:
            print("\n[pose] 正在保存并退出…", flush=True)
        finally:
            stop.set()
            if inspection:
                (session/"detection_diagnostics.json").write_text(json.dumps(diagnostics.summary(), indent=2, ensure_ascii=False)+"\n")
    if not frames:
        raise RuntimeError("No input frames received")
    summary = {"frames": frames, "valid_frames": valid_counts, "dropped_pairs": capture.dropped if capture else 0,
               "synthetic_only": bool(args.demo)}
    if errors:
        summary["synthetic_error"] = {"position_mean_m": float(np.mean([e["position_m"] for e in errors])),
                                      "position_max_m": float(np.max([e["position_m"] for e in errors])),
                                      "rotation_mean_deg": float(np.mean([e["rotation_deg"] for e in errors])),
                                      "rotation_max_deg": float(np.max([e["rotation_deg"] for e in errors]))}
    if calibrators:
        calibrated, failures = {}, {}
        for name, calibrator in calibrators.items():
            try:
                calibrated[name] = calibrator.finish()
                calibrated[name]["session"] = str(session)
                calibrated[name]["calibrated_at"] = datetime.now().astimezone().isoformat()
            except ValueError as exc:
                failures[name] = str(exc)
        (session/"wrist_calibration_report.json").write_text(json.dumps({"targets": calibrated, "failures": failures, "observations": diagnostics.summary()}, indent=2, ensure_ascii=False)+"\n")
        if failures:
            details = "；".join(f"{name}: {entry['hint']}" for name, entry in diagnostics.summary()['targets'].items())
            raise ValueError(f"标定未通过，已有参数未修改。{details} 详细原因：{session/'detection_diagnostics.json'}；标签对："+json.dumps(failures, ensure_ascii=False))
        save_calibrations(cfg["bundle_calibration"], cfg, calibrated)
        summary["wrist_calibration"] = cfg["bundle_calibration"]
        print(f"[calibrate] 已保存 {cfg['bundle_calibration']}", flush=True)
    if args.command == "check":
        summary["observations"] = diagnostics.summary()
        for name, entry in diagnostics.summary()["targets"].items():
            print(f"[check] {name}: {entry['hint']}", flush=True)
    metadata["finished"] = True
    (session/"session.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+"\n")
    (session/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[saved] {session}", flush=True)
    return summary
