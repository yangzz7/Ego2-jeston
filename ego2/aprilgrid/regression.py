"""Replay the original JPEGs from an AprilGrid session, without a camera.

python -m ego2.aprilgrid.regression SESSION --output NEW_DIRECTORY [--save-rrd]
The optional Rerun reader needs requirements-aprilgrid-replay.txt. Normal live
tracking and fixture tests do not depend on it. Source sessions are read-only.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from ..common.config import load_calibration
from ..common.sources import Pair
from .app import frame_record
from .config import Board
from .temporal import Pipeline
from .tracking import configure_tracking, display_source
from .camera_pose import configure_pose_viewer, camera_pose_frame, POSE_METADATA
from .stereo_check import StereoCheck, print_report, report_overview


def images_from_session(session, rows, cache=None):
    if cache is not None:
        for row in rows:
            images = tuple(cv2.imread(str(cache/f"{row['frame']}-{name}.jpg")) for name in ("cam0", "cam1"))
            if any(image is None for image in images):
                raise ValueError(f"Missing cached stereo pair {row['frame']}")
            yield row, images
        return
    try:
        from datafusion import col, lit
        from rerun.server import Server
    except ImportError as exc:
        raise RuntimeError("RRD replay needs: pip install -r requirements-aprilgrid-replay.txt") from exc
    with Server(host="127.0.0.1", datasets={"replay": [str(session/"viewer.rrd")]}) as server:
        view = server.client().get_dataset("replay").filter_contents(["/raw/**"])
        # Bound Arrow/Python memory; never materialize the entire recording.
        for start in range(0, len(rows), 32):
            part = rows[start:start+32]
            wanted = {r["frame"] for r in part}
            decoded = {}
            predicate = col("frame") == lit(part[0]["frame"])
            for row in part[1:]:
                predicate = predicate | (col("frame") == lit(row["frame"]))
            reader = view.reader(index="frame").filter(predicate)
            for batch in reader.collect():
                for i, frame in enumerate(batch.column("frame").to_pylist()):
                    if frame not in wanted:
                        continue
                    images = []
                    for name in ("cam0", "cam1"):
                        blob = batch.column(f"/raw/{name}/image:EncodedImage:blob")[i].as_py()
                        if not blob:
                            raise ValueError(f"Missing {name} image at frame {frame}")
                        images.append(cv2.imdecode(np.frombuffer(bytes(blob[0]), np.uint8), cv2.IMREAD_COLOR))
                    if any(image is None for image in images):
                        raise ValueError(f"Invalid JPEG at frame {frame}")
                    decoded[frame] = tuple(images)
            for row in part:
                if row["frame"] not in decoded:
                    raise ValueError(f"Missing image pair at frame {row['frame']}")
                yield row, decoded.pop(row["frame"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New directory; never overwrite")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--save-rrd", action="store_true")
    parser.add_argument("--frames-cache", type=Path, help="Optional FRAME-cam0/cam1.jpg directory")
    args = parser.parse_args(argv)
    if args.stride < 1:
        parser.error("stride must be positive")
    cv2.setNumThreads(1)
    session = args.session.expanduser().resolve()
    metadata = json.loads((session/"session.json").read_text())
    cfg = metadata["config"]
    configure_tracking(cfg["tracking"])
    configure_pose_viewer(cfg["viewer"])
    cameras = load_calibration(session/"calibration.yaml")
    board = Board(yaml.safe_load((session/"target.yaml").read_text()))
    stereo_check = StereoCheck(cfg, cameras, board, synthetic_only=metadata.get("synthetic_only", False),
                               calibration_file=session/"calibration.yaml")
    rows = [json.loads(line) for line in (session/"poses.jsonl").read_text().splitlines()][::args.stride]
    if not rows:
        parser.error("No recorded frames")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("calibration.yaml", "target.yaml"):
        shutil.copy2(session/name, args.output/name)
    metadata.update(finished=False, replay_source=str(session), replay_stride=args.stride,
                    replay_input="Recorded JPEGs, not original lossless camera pixels",
                    display_isp={"active": False, "algorithm_input_unchanged": True,
                                 "reason": "Replay uses unenhanced /raw RGB images; no original Bayer transport"},
                    camera_pose_readout=POSE_METADATA)
    metadata_path = args.output/"session.json"
    metadata_path.write_text(json.dumps(metadata, indent=2)+"\n")
    statuses, elapsed, detection, solve, errors = Counter(), [], [], [], []
    previous_valid = gained = lost = 0
    display_frames = predicted_frames = decode_frames = 0
    with ExitStack() as resources:
        pipeline = Pipeline(cfg, cameras, board)
        resources.callback(pipeline.close)
        solver = pipeline.solver
        viewer = None
        if args.save_rrd:
            from .visualizer import Visualizer
            viewer = Visualizer(cfg, cameras, board, SimpleNamespace(headless=True, connect=None, web=False,
                                web_port=9090, grpc_port=9876), args.output/"viewer.rrd")
            resources.callback(viewer.close)
        table = resources.enter_context((args.output/"poses.jsonl").open("w"))
        stream = images_from_session(session, rows, args.frames_cache)
        resources.callback(stream.close)
        for row, images in stream:
            begin = time.perf_counter()
            obs = pipeline.process(images, row["time_s"])
            track = pipeline.track
            camera_pose = camera_pose_frame(row["time_s"], track, solver.transforms, cfg["viewer"]["pose_hold_s"])
            stereo_check.observe(row["time_s"], row["frame"], pipeline.decoded_observations, camera_pose)
            ms = (time.perf_counter()-begin)*1000
            elapsed.append(ms)
            detection.append(pipeline.detection_ms)
            solve.append(pipeline.solve_ms)
            display_frames += any(display_source(o, track) is not None for o in obs)
            predicted_frames += track.display_predicted
            decode_frames += solver.diagnostics["temporal"]["decoded_this_frame"]
            statuses[track.status] += 1
            accepted, was_accepted = track.measurement is not None, row["T_reference_board"] is not None
            previous_valid += was_accepted
            gained += accepted and not was_accepted
            lost += was_accepted and not accepted
            if accepted:
                errors.append(track.measurement.reprojection_px)
            pair = Pair(images, row["source_time_s"], 0, row["frame"])
            table.write(json.dumps(frame_record(pair, row["time_s"], obs, track, board, solver, ms, camera_pose), allow_nan=False)+"\n")
            if viewer:
                viewer.log(pair, row["time_s"], obs, track, ms, solver.diagnostics, camera_pose)
            if len(elapsed) % 100 == 0:
                print(f"[replay] {len(elapsed)}/{len(rows)} | valid={len(errors)}", flush=True)
    print(f"[stereo-check] 回放已结束，正在独立求解 {len(stereo_check.samples)} 组双目角点…", flush=True)
    stereo_report = stereo_check.finish(progress=print)
    (args.output/"stereo_check.json").write_text(json.dumps(stereo_report, indent=2, allow_nan=False)+"\n")
    summary = {"frames": len(rows), "valid_pose_frames": len(errors), "valid_fraction": len(errors)/len(rows),
               "display_pose_frames": display_frames, "predicted_display_frames": predicted_frames,
               "full_detection_frames": decode_frames,
               "status_counts": dict(statuses), "source_session": str(session),
               "recorded_previous_valid": previous_valid, "gained": gained, "lost": lost,
               "processing_ms_mean": float(np.mean(elapsed)), "processing_ms_p95": float(np.percentile(elapsed, 95)),
               "detection_ms_mean": float(np.mean(detection)), "solve_ms_mean": float(np.mean(solve)),
               "reprojection_px_mean": float(np.mean(errors)) if errors else None,
               "stereo_check": report_overview(stereo_report),
               "note": "JPEG replay; previous recorded statuses used lossless live pixels. No real pose ground truth."}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    metadata["finished"] = True
    metadata_path.write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
    print_report(stereo_report)
    print(f"[stereo-check] 完整对照与独立角点样本：{args.output/'stereo_check.json'}", flush=True)


if __name__ == "__main__":
    main()
