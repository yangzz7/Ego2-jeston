"""Independent capture → detect → joint pose → synchronized Rerun loop."""
from contextlib import ExitStack
from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import shutil
import threading
import time

import numpy as np
import yaml

from ..common.paths import OUTPUT
from ..common.runtime import FrameClock, LatestPairs
from ..common.sources import Source
from .pose import rotation_distance
from .temporal import Pipeline
from .tracking import display_source
from .camera_pose import camera_pose_frame, POSE_METADATA
from .stereo_check import StereoCheck, print_report, report_overview
from .display_isp import display_isp_active, create_visualizer


def frame_record(pair, stamp, observations, track, board, solver, processing_ms, camera_pose=None):
    m = track.measurement
    if camera_pose is None:
        camera_pose = camera_pose_frame(stamp, track, solver.transforms, solver.cfg["viewer"].get("pose_hold_s", .3))
    tags = []
    for obs in observations:
        accepted = m is not None and obs.key in m.inliers
        shown = display_source(obs, track)
        tag = {"camera": obs.camera, "id": obs.marker_id, "pixels": obs.pixels.tolist(),
               "valid_pose": accepted, "status": (m.mode if accepted else
                   obs.reason if obs.reason != "ok" else track.status if track.status == "pose_jump" else obs.pose_reason),
               "corner_refinement": obs.refinement_status,
               "corner_shift_px": obs.refinement_shift_px,
               "observation_source": obs.source, "flow_error_px": obs.flow_error_px,
               "appearance_score": obs.appearance_score, "appearance_valid": bool(obs.appearance_valid),
               "confirmation_age_s": obs.confirmation_age_s,
               "used_for_pose": obs.source != "board" and obs.reason == "ok",
               "displayed": shown is not None, "display_pose_source": shown}
        if obs.reprojection_px is not None:
            tag["reprojection_px"] = obs.reprojection_px
        if accepted:
            pose = board.tag_pose(m.pose, obs.marker_id)
            tag["T_reference_tag"] = pose.tolist()
            tag["T_camera_tag"] = (solver.transforms[obs.camera] @ pose).tolist()
        if shown is not None:
            pose = board.tag_pose(track.display_pose, obs.marker_id)
            tag["T_display_reference_tag"] = pose.tolist()
            tag["T_display_camera_tag"] = (solver.transforms[obs.camera] @ pose).tolist()
            tag["display_pixels"] = (obs.display_pixels if obs.display_pixels is not None else obs.pixels).tolist()
        tags.append(tag)
    return {"time_s": stamp, "source_time_s": pair.stamp, "frame": pair.sequence,
            "status": track.status, "processing_ms": processing_ms,
            "solver": solver.diagnostics,
            "T_reference_board": m.pose.tolist() if m else None,
            "display_T_reference_board": track.display_pose.tolist() if track.display_pose is not None else None,
            "display_predicted": track.display_predicted, "prediction_age_s": track.prediction_age_s,
            "reprojection_px": m.reprojection_px if m else None, "tags": tags,
            "camera_pose": camera_pose,
            **{f"T_board_{name}": value["T_board_camera"] if value else None
               for name, value in camera_pose["cameras"].items()}}


def run(cfg, cameras, board, args, *, viewer_factory=None, stop_event=None):
    isp_active = display_isp_active(cfg, args)
    if isp_active:
        # Preserve transport for the display branch without changing the normal
        # decode/white-balance path or its images consumed by Pipeline.
        cfg["input"]["preserve_raw"] = True
    stereo_check = StereoCheck(cfg, cameras, board, synthetic_only=bool(args.demo))
    session = None
    metadata = None
    if not args.no_record:
        session = (Path(args.session).expanduser().resolve() if args.session else
                   OUTPUT/"aprilgrid"/datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        session.mkdir(parents=True, exist_ok=False)
        shutil.copy2(cfg["target"], session/"target_source.yaml")
        (session/"target.yaml").write_text(yaml.safe_dump(board.data, sort_keys=False))
        shutil.copy2(cfg["calibration"]["file"], session/"calibration.yaml")
        if isp_active:
            shutil.copy2(cfg["viewer"]["display_isp"]["config"], session/"display_isp.yaml")
        metadata = {"version": 1, "finished": False, "synthetic_only": bool(args.demo),
            "reference_camera": cfg["reference"], "units": "metres",
            "board_frame": "grid centre; X print-right, Y print-up, Z out of printed face",
            "tag_frame": "black-square centre; axes parallel to board frame",
            "camera_frame": "X right, Y down, Z forward",
            "transforms": "T_A_B maps B coordinates into A; T_board_cam0/cam1 are optical camera poses in the board frame; no VIO",
            "camera_pose_readout": POSE_METADATA,
            "display_isp": {"active": isp_active, "algorithm_input_unchanged": True,
                "input": "original packed Bayer; never reconstructed from RGB",
                "raw_rrd": "/raw/cam0/image, /raw/cam1/image: every inference frame",
                "display_rrd": "/display and /tracking: same-frame ISP images, overlays and pose row; may skip display frames"},
            "clock": "host receipt after capture read, before decoding; not exposure time" if not args.demo and cfg["input"]["type"] == "cyper" else "media time",
            "recording": "processed complete pairs only; live inference may skip pairs; not a lossless camera archive",
            "config": cfg}
        if cfg.get('unified_preview'):
            metadata['display_isp'].update(raw_rrd=None,
                display_rrd='parent viewer.rrd; /cameras/<role>/<cam>; bounded complete-frame preview')
        (session/"session.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+"\n")
        print(f"[session] {session}", flush=True)
    print(f"[aprilgrid] {board.rows}x{board.cols}, tag={board.size*1000:g} mm, border={board.border}, reference={cfg['reference']}", flush=True)
    if args.demo:
        print("[demo] 合成鱼眼图像经过真实识码与位姿求解；不代表实机精度。", flush=True)
    stop = stop_event if stop_event is not None else threading.Event()
    clock, capture = FrameClock(), None
    frames, valid_frames, mono_frames, detections = 0, 0, 0, 0
    display_frames = predicted_frames = decode_frames = 0
    tracked_tags = displayed_tags = 0
    processing, errors, reprojections = [], [], []
    detection_times, solve_times, viewer_times = [], [], []
    decode_times, flow_times, recovery_times = [], [], []
    statuses = Counter()
    with ExitStack() as resources:
        pipeline = Pipeline(cfg, cameras, board, throttle_empty=True)
        resources.callback(pipeline.close)
        solver = pipeline.solver
        viewer = None
        if not args.no_viz:
            viewer = create_visualizer(cfg, cameras, board, args, session/"viewer.rrd" if session else None,
                                       viewer_factory=viewer_factory)
            resources.callback(viewer.close)
        table = resources.enter_context((session/"poses.jsonl").open("w")) if session else None
        if args.demo:
            from .demo import DemoSource
            source = DemoSource(cfg, cameras, board)
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
        last_report = started-2
        try:
            while not stop.is_set():
                if (args.frames and frames >= args.frames) or (args.seconds and time.monotonic()-started >= args.seconds):
                    break
                pair = read()
                if pair is None:
                    break
                stamp, _ = clock.update(pair.stamp)
                begin = time.perf_counter()
                observations = pipeline.process(pair.images, stamp)
                track = pipeline.track
                camera_pose = camera_pose_frame(stamp, track, solver.transforms, cfg["viewer"].get("pose_hold_s", .3))
                stereo_check.observe(stamp, pair.sequence, pipeline.decoded_observations, camera_pose)
                ms = (time.perf_counter()-begin)*1000
                processing.append(ms)
                detection_times.append(pipeline.detection_ms)
                solve_times.append(pipeline.solve_ms)
                decode_times.append(pipeline.decode_ms)
                flow_times.append(pipeline.flow_ms)
                recovery_times.append(pipeline.recovery_ms)
                shown = sum(display_source(o, track) is not None for o in observations)
                display_frames += shown > 0
                displayed_tags += shown
                predicted_frames += track.display_predicted
                decode_frames += solver.diagnostics["temporal"]["decoded_this_frame"]
                statuses[track.status] += 1
                detections += solver.diagnostics["temporal"]["decoded_tags"]
                tracked_tags += solver.diagnostics["temporal"]["flow_tags"]
                if track.measurement is not None:
                    valid_frames += 1
                    mono_frames += track.measurement.mode == "mono"
                    reprojections.append(track.measurement.reprojection_px)
                    if args.demo:
                        errors.append([float(np.linalg.norm(track.measurement.pose[:3, 3]-source.truth[:3, 3])),
                            float(np.rad2deg(rotation_distance(track.measurement.pose[:3, :3], source.truth[:3, :3])))])
                if table:
                    table.write(json.dumps(frame_record(pair, stamp, observations, track, board, solver, ms, camera_pose), allow_nan=False)+"\n")
                if viewer:
                    log_started = time.perf_counter()
                    viewer.log(pair, stamp, observations, track, ms, solver.diagnostics, camera_pose)
                    viewer_times.append((time.perf_counter()-log_started)*1000)
                frames += 1
                now = time.monotonic()
                if now-last_report >= 1:
                    counts = {ch["camera"]: sum(o.camera == ch["camera"] for o in observations) for ch in cfg["channels"]}
                    support = solver.diagnostics
                    print(f"[aprilgrid] frames={frames} | {track.status} | tags={counts} | "
                          f"inliers={support['inliers']}/{support['observations']} | detect+solve={ms:.1f} ms | "
                          f"decode={pipeline.decode_ms:.1f} flow={pipeline.flow_ms:.1f} "
                          f"solve={pipeline.solve_ms:.1f} overlays={pipeline.recovery_ms:.1f} "
                          f"preview={viewer_times[-1] if viewer_times else 0:.1f} ms", flush=True)
                    if getattr(args, 'metrics', None):
                        metrics_path = Path(args.metrics)
                        pending = metrics_path.with_suffix('.tmp')
                        pending.write_text(json.dumps(dict(frames=frames, status=track.status,
                            decode_ms=pipeline.decode_ms, flow_ms=pipeline.flow_ms,
                            solve_ms=pipeline.solve_ms, overlays_ms=pipeline.recovery_ms,
                            preview_ms=viewer_times[-1] if viewer_times else 0)))
                        pending.replace(metrics_path)
                    if table:
                        table.flush()
                    last_report = now
        except KeyboardInterrupt:
            print("\n[aprilgrid] 正在保存并释放相机…", flush=True)
        finally:
            stop.set()
        elapsed = time.monotonic()-started
    if not frames:
        raise RuntimeError("No input frames received")
    print(f"[stereo-check] 采集已结束，正在独立求解 {len(stereo_check.samples)} 组双目角点…", flush=True)
    stereo_report = stereo_check.finish(progress=print)
    summary = {"frames": frames, "valid_pose_frames": valid_frames, "mono_frames": mono_frames,
        "display_pose_frames": display_frames, "predicted_display_frames": predicted_frames,
        "full_detection_frames": decode_frames,
        "detected_tags_per_eye_mean": detections/(2*frames),
        "tracked_tags_per_eye_mean": tracked_tags/(2*frames),
        "displayed_tags_per_eye_mean": displayed_tags/(2*frames),
        "processed_fps": frames/elapsed, "processing_ms_mean": float(np.mean(processing)),
        "processing_ms_p95": float(np.percentile(processing, 95)),
        "detection_ms_mean": float(np.mean(detection_times)),
        "solve_ms_mean": float(np.mean(solve_times)),
        "viewer_ms_mean": float(np.mean(viewer_times)) if viewer_times else None,
        "decode_ms_mean": float(np.mean(decode_times)),
        "flow_ms_mean": float(np.mean(flow_times)),
        "display_recovery_ms_mean": float(np.mean(recovery_times)),
        "display_isp": viewer.statistics() if isp_active and viewer else {"enabled": False},
        "status_counts": dict(statuses),
        "dropped_pairs": capture.dropped if capture else 0, "synthetic_only": bool(args.demo),
        "stereo_check": report_overview(stereo_report),
        "reprojection_px_mean": float(np.mean(reprojections)) if reprojections else None}
    if errors:
        summary["synthetic_error"] = {"position_mean_m": float(np.mean(errors, axis=0)[0]),
                                      "rotation_mean_deg": float(np.mean(errors, axis=0)[1])}
    if session:
        (session/"stereo_check.json").write_text(json.dumps(stereo_report, indent=2, allow_nan=False)+"\n")
        (session/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        metadata["finished"] = True
        (session/"session.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+"\n")
        print(f"[saved] {session}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print_report(stereo_report)
    if session:
        print(f"[stereo-check] 完整对照与独立角点样本：{session/'stereo_check.json'}", flush=True)
    return summary
