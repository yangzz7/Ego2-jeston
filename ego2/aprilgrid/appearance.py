"""Batch identical local bit checks; never search for or reassign tag IDs."""
from collections import defaultdict

import cv2
import numpy as np


def verify(pipeline, observations, gray):
    groups = defaultdict(list)
    for obs in observations:
        obs.appearance_valid = obs.appearance_weak = False
        obs.appearance_score = None
        groups[(obs.camera, obs.view_index)].append(obs)
    # cv2.remap requires both dimensions below SHRT_MAX. Bound scratch maps
    # even when a custom board contains far more tags than the usual 6x6.
    batch_size = max(1, 30000//len(pipeline.samples))
    batches = [(key, group[start:start+batch_size]) for key, group in groups.items()
               for start in range(0, len(group), batch_size)]
    for (name, index), group in batches:
        cam = pipeline.cameras[name]
        uv = []
        for obs in group:
            h = cv2.getPerspectiveTransform(pipeline.square, obs.view_pixels.astype(np.float32))
            uv.append(cv2.perspectiveTransform(pipeline.samples, h).reshape(-1, 2))
        rotation = pipeline.detector.views[name][1][index][4]
        raw, valid = pipeline.detector.to_raw(cam, np.concatenate(uv), group[0].K, rotation)
        samples = len(pipeline.samples)
        usable = (valid & np.all(raw >= 0, axis=1) & np.all(raw < np.array(cam.size)-1, axis=1))
        usable = usable.reshape(len(group), samples).all(axis=1)
        mx, my = raw[:, :1].astype(np.float32), raw[:, 1:].astype(np.float32)
        mask = cv2.remap(pipeline.detector.native_masks[name], mx, my, cv2.INTER_LINEAR)
        usable &= (mask.reshape(len(group), samples) >= .999).all(axis=1)
        values = cv2.remap(gray, mx, my, cv2.INTER_LINEAR).reshape(len(group), -1, 4).mean(axis=2)
        # Each marker has its own black/white pattern and medians; padding and
        # reducing across tags would change the appearance acceptance rule.
        for obs, cells, ok in zip(group, values, usable):
            if not ok:
                continue
            white = pipeline.patterns[obs.marker_id]
            dark, light = np.median(cells[~white]), np.median(cells[white])
            if light-dark < pipeline.settings['appearance_min_contrast']:
                continue
            mismatch = (cells > (light+dark)/2) != white
            errors = int(mismatch[pipeline.inner].sum())
            obs.appearance_score = float(1-mismatch.mean())
            limit = pipeline.settings['appearance_max_bit_errors']
            border_error = mismatch[~pipeline.inner].mean()
            obs.appearance_valid = bool(errors <= limit and border_error <= .15)
            obs.appearance_weak = bool(errors <= limit+2 and border_error <= .2)
