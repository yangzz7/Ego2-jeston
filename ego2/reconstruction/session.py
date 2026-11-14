"""Durable keyframe depth cache and pose updates, separated from live preview."""
from pathlib import Path
import json
import cv2
import numpy as np

from .depth import StereoDepth


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def initialization_complete(snapshot, map_, *, allow_legacy=False):
    if snapshot.get("mode") in ("stereo", "demo"):
        return True
    # Version 1 recordings predate the explicit completion flag. Preserve their
    # offline export behavior; new native sessions must carry positive proof.
    complete = map_.get("imu_initialization_complete", allow_legacy and snapshot.get("version", 1) < 2)
    return bool(map_.get("imu_initialized") and complete)


def eligible_keyframes(snapshot, *, allow_legacy=False):
    result = {}
    for map_ in snapshot.get("maps", []):
        if not initialization_complete(snapshot, map_, allow_legacy=allow_legacy):
            continue
        for kf in map_["keyframes"]:
            if kf.get("image_available"):
                # Globally keyed by image timestamp; an atlas merge can move a
                # keyframe between map IDs without copying its depth/image data.
                result[kf["image_key"]] = {**kf, "map_id": map_["id"]}
    return result


class DenseMap:
    def __init__(self, cfg, session, viewer=None):
        self.cfg, self.session, self.viewer = cfg, Path(session), viewer
        self.stereo = StereoDepth(cfg)
        self.live = {}  # image_key -> (map_id, pose); point arrays belong to Rerun
        self.ready = False
        self.processed = 0
        self.stats = {}
        (self.session/"depth").mkdir(exist_ok=True)
        write_json(self.session/"rectification.json", self.stereo.metadata())

    def depth(self, key):
        path = self.session/"depth"/(key+".npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                return data["depth"], data["color"]
        base = self.session/"keyframes"/key
        images = [cv2.imread(str(base)+f"_{name}.jpg") for name in ("cam1","cam0")]
        depth, color = self.stereo.compute(*images)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, depth=depth, color=color)
        tmp.replace(path)
        self.processed += 1
        return depth, color

    def prepare_export(self, snapshot):
        # An inactive, previously initialized map can still be exported while
        # the live active map is waiting for a new initialization.
        for key in eligible_keyframes(snapshot, allow_legacy=True):
            self.depth(key)

    def update(self, snapshot, stamp, budget=1):
        active = next((m for m in snapshot.get("maps", []) if m["id"] == snapshot["active_map"]), None)
        self.ready = active is not None and initialization_complete(snapshot, active)
        candidates = eligible_keyframes(snapshot) if self.ready else {}
        ordered = sorted(candidates, key=lambda k: candidates[k]["timestamp"])
        live_keys = set(ordered[-self.cfg["depth"]["max_live_keyframes"]:])
        # Only display the active map: unconnected atlas maps have independent
        # origins. Export keeps them separate instead of falsely overlaying them.
        live_keys = {k for k in live_keys if candidates[k]["map_id"] == snapshot["active_map"]}
        if self.viewer:
            self.viewer.time(stamp)
        for key in list(self.live):
            kf = candidates.get(key)
            if key not in live_keys or kf["map_id"] != self.live[key][0]:
                if self.viewer:
                    self.viewer.remove_chunk(self.live[key][0], key)
                del self.live[key]
        # Gate computation as well as visualization. In particular, shutdown's
        # unlimited budget must not bypass initialization or resurrect caches.
        if not self.ready:
            self.write_status(snapshot, candidates)
            return
        # Existing point data never gets irreversibly fused at stale poses.
        for key, (map_id, old) in list(self.live.items()):
            pose = np.asarray(candidates[key]["T_world_camera"])
            if not np.allclose(pose,old,atol=1e-6):
                if self.viewer:
                    self.viewer.pose(map_id,key,pose)
                self.live[key] = (map_id,pose)
        # Pause NEW live integration while current tracking is lost.
        if budget is not None and snapshot.get("state") != 2:
            self.write_status(snapshot, candidates)
            return
        pending = [k for k in ordered if not (self.session/"depth"/(k+".npz")).exists()]
        if budget is not None:
            # Prefer recent observations to keep live visualization responsive.
            pending = pending[::-1][:budget]
        for key in pending:
            self.depth(key)
        for key in ordered:
            if key in self.live or key not in live_keys or not (self.session/"depth"/(key+".npz")).exists():
                continue
            pose = np.asarray(candidates[key]["T_world_camera"])
            depth,color = self.depth(key)
            points,colors = self.stereo.points(depth,color)
            if self.viewer:
                self.viewer.chunk(candidates[key]["map_id"],key,pose,points,colors)
            self.live[key] = (candidates[key]["map_id"],pose)
            self.stats[key] = len(points)
        self.write_status(snapshot, candidates)

    def write_status(self, snapshot, candidates):
        write_json(self.session/"dense_status.json", {
            "initialization_complete": self.ready,
            "reconstruction_state": "waiting_for_vio" if not self.ready else
                                    "tracking_paused" if snapshot.get("state") != 2 else "reconstructing",
            "eligible_keyframes": len(candidates), "depth_keyframes": len(list((self.session/"depth").glob("*.npz"))),
            "active_map": snapshot.get("active_map"), "live_keyframes": len(self.live),
            "live_points": sum(self.stats.get(k,0) for k in self.live),
            "final": bool(snapshot.get("final")),
        })
