#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ -x "$ROOT/.venv-export/bin/python" ]] || { echo 'The optional export environment was not installed.' >&2; exit 2; }
export EGO2_PYTHON="$ROOT/.venv-jetson/bin/python"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
exec "$ROOT/run_reconstruction.sh" export --open3d-python "$ROOT/.venv-export/bin/python" "$@"
