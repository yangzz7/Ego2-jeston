#!/usr/bin/env python3
"""Fetch the exact upstream assets used by the Jetson source snapshot."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "gesture": {
        "path": "models/gesture_recognizer.task",
        "url": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
        "sha256": "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482",
        "size": 8373440,
    },
    "vocabulary": {
        "path": "vocabulary/ORBvoc.txt",
        "url": "https://raw.githubusercontent.com/UZ-SLAMLab/ORB_SLAM3/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4/Vocabulary/ORBvoc.txt.tar.gz",
        "sha256": "f8dd027f7a6cb88129821341194d7f2c75b77b3394257ddd0d2229863d1a3570",
        "size": 145250924,
        "member": "ORBvoc.txt",
    },
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def matches(path, asset):
    return path.is_file() and path.stat().st_size == asset["size"] and digest(path) == asset["sha256"]


def fetch(root, name, check_only=False):
    asset = ASSETS[name]
    target = root / asset["path"]
    if matches(target, asset):
        print(f"Verified: {asset['path']}", flush=True)
        return
    if check_only:
        raise RuntimeError(f"Missing or checksum mismatch: {asset['path']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Stage on the destination filesystem; keep any previous file until the
    # complete replacement has the expected size and SHA-256.
    with tempfile.TemporaryDirectory(prefix=".asset-", dir=target.parent) as folder:
        download = Path(folder) / "download"
        print(f"Downloading: {asset['path']}", flush=True)
        request = urllib.request.Request(asset["url"], headers={"User-Agent": "Ego2-Jetson-asset-fetcher"})
        with urllib.request.urlopen(request, timeout=60) as response, download.open("wb") as out:
            count = 0
            for block in iter(lambda: response.read(1024 * 1024), b""):
                count += len(block)
                if count > 200 * 1024 * 1024:
                    raise RuntimeError("Unexpectedly large asset response")
                out.write(block)
        prepared = download
        if "member" in asset:
            prepared = Path(folder) / "verified-content"
            with tarfile.open(download, "r:gz") as archive:
                members = [m for m in archive.getmembers() if m.isfile() and m.name.removeprefix("./") == asset["member"]]
                if len(members) != 1 or members[0].size != asset["size"]:
                    raise RuntimeError("Unexpected vocabulary archive contents")
                # Read one regular member; never extract archive paths or links.
                with archive.extractfile(members[0]) as source, prepared.open("wb") as out:
                    shutil.copyfileobj(source, out)
        if not matches(prepared, asset):
            raise RuntimeError(f"Downloaded asset checksum mismatch: {asset['path']}")
        os.replace(prepared, target)
    print(f"Verified and installed: {asset['path']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", choices=["all", *ASSETS], default="all")
    parser.add_argument("--root", type=Path, default=ROOT, help="Destination project root")
    parser.add_argument("--check", action="store_true", help="Verify existing files without downloading")
    args = parser.parse_args()
    try:
        for name in ASSETS if args.asset == "all" else [args.asset]:
            fetch(args.root, name, args.check)
    except (OSError, RuntimeError, tarfile.TarError) as exc:
        parser.exit(1, f"Asset error: {exc}\n")


if __name__ == "__main__":
    main()
