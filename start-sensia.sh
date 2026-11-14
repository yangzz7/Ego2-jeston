#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$ROOT/.venv-jetson/bin/python" ]]; then
  echo 'Install this bundle first. This launcher must live in the installed Ego2 directory.' >&2; exit 2
fi
export EGO2_PYTHON="$ROOT/.venv-jetson/bin/python"
export PATH="$ROOT/.venv-jetson/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/build/reconstruction/lib:$ROOT/.native/pangolin/lib:$ROOT/.native/pangolin/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="$ROOT/.native/pangolin${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
cd "$ROOT"
exec "$ROOT/run_multi_vio.sh" "$@"
