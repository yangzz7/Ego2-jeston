"""Storage policy shared by the four unified modes (no camera dependencies)."""
from datetime import datetime
from pathlib import Path
import json
import shutil

from ..common.paths import OUTPUT

FOLDERS = dict(vio='trajectory', multi_vio='trajectory', gesture='gesture',
               aprilgrid='aprilgrid', reconstruction='reconstruction')


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def new_session(mode, requested=None):
    for name in set(FOLDERS.values()):
        (OUTPUT / name).mkdir(parents=True, exist_ok=True)
    if requested:
        root = Path(requested).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=False)
    else:
        while True:
            root = OUTPUT / FOLDERS[mode] / datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
            try:
                root.mkdir()
                break
            except FileExistsError:
                continue
    return root


def status(root, state, **changes):
    path = Path(root) / 'session.json'
    metadata = json.loads(path.read_text()) if path.exists() else {}
    metadata.update(changes, status=state, finished=state == 'complete')
    if 'created_at' not in metadata:
        metadata['created_at'] = datetime.now().astimezone().isoformat()
    if state in ('complete', 'incomplete'):
        metadata['ended_at'] = datetime.now().astimezone().isoformat()
    write_json(path, metadata)
    return metadata


def validate_rrd(path):
    # Read the complete stream: a file's existence alone doesn't prove it can play.
    from rerun.experimental import RrdReader
    count = sum(1 for chunk in RrdReader(str(path)).stream()
                if (chunk.entity_path.startswith('/cameras/') and chunk.entity_path.endswith(('/image', '/rgb')))
                or chunk.entity_path in ('/rgb/cam0', '/rgb/cam1'))
    if not count:
        raise RuntimeError(f'Empty Rerun replay: {path}')


def finish_replay(root, source, work):
    root, source = Path(root), Path(source)
    destination = root / 'replay.rrd'
    validate_rrd(source)
    if source != destination:
        shutil.move(str(source), destination)
    status(root, 'complete', format='rerun', replay_file='replay.rrd')
    if work and Path(work).exists():
        shutil.rmtree(work)
