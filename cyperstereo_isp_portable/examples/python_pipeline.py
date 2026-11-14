#!/usr/bin/env python3
"""Process saved C1/C2 RAW8 files with a persistent ISP; opens no camera.

From the package root:
    python examples/python_pipeline.py C1_bayer.raw8 C2_bayer.raw8 --output result

Each input must be an unmodified 1280x1024 RAW8 plane including its metadata
row, such as camera_tuner's c1_bayer.raw8 / c2_bayer.raw8 snapshots. The script
writes c1_bgr.npy and c2_bgr.npy (NumPy alone); --png additionally needs cv2.

Integration at an existing Ego2 capture loop (create Isp once, outside loop):

    with Isp(config="configs/s200032.yaml") as isp:
        while running:
            ok, packed = cap.read()  # existing native YUYV-labelled transport
            if not ok:
                raise RuntimeError("camera read failed")
            c1_bgr, c2_bgr = isp.process_transport(packed)
            # Replaces BOTH decode_cyper(...) AND white_balance(...).
            # Existing detection sees BGR. For Rerun, convert BGR->RGB once:
            # rr.log("cam0", rr.Image(c1_bgr[:, :, ::-1].copy()))

Keep the existing capture configured with CAP_PROP_CONVERT_RGB=0 and native
1280x1024 transport. Do not run another white-balance or gamma stage after
the ISP, and do not rebuild Isp each frame (that loses temporal AWB state).
Physical left/right assignment still requires covering each lens to confirm.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from cyperstereo_isp import Isp  # noqa: E402


def read_raw8(filename: Path) -> np.ndarray:
    expected = 1024 * 1280
    if filename.stat().st_size != expected:
        raise ValueError(f"{filename}: expected exactly {expected} RAW8 bytes including metadata row")
    return np.fromfile(filename, dtype=np.uint8).reshape(1024, 1280)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("c1_raw", type=Path)
    parser.add_argument("c2_raw", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path, default=Path("isp-output"))
    parser.add_argument("--png", action="store_true", help="Also save PNG with optional OpenCV Python")
    args = parser.parse_args()
    try:
        cv2 = None
        if args.png:
            try:
                import cv2
            except ImportError as error:
                raise RuntimeError("--png requires OpenCV Python; omit it to save NumPy arrays only") from error
        packed = np.stack((read_raw8(args.c1_raw), read_raw8(args.c2_raw)), axis=2)
        with Isp(config=args.config, library=args.library) as isp:
            c1, c2 = isp.process_transport(packed)
            gains = isp.last_white_balance_rgb
        # A new output directory prevents accidental replacement of user data.
        args.output.mkdir(parents=True, exist_ok=False)
        for name, image in (("c1", c1), ("c2", c2)):
            with (args.output / f"{name}_bgr.npy").open("xb") as stream:
                np.save(stream, image, allow_pickle=False)
            if cv2 is not None and not cv2.imwrite(str(args.output / f"{name}_color.png"), image):
                raise RuntimeError(f"Could not save {name} PNG")
        print(f"Saved full-resolution BGR8 arrays to {args.output.resolve()}")
        print(f"Actual white balance [C1/C2, R/G/B]: {gains.tolist() if gains is not None else None}")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
