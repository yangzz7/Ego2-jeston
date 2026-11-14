#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -d "$ROOT/.venv-jetson" ]]; then
  echo 'Run the installed Ego2/check-jetson.sh after deployment.' >&2; exit 2
fi
export PATH="$ROOT/.venv-jetson/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/.native/pangolin/lib:$ROOT/.native/pangolin/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
export MPLBACKEND=Agg
cd "$ROOT"
"$ROOT/.venv-jetson/bin/python" "$ROOT/.deployment/scripts/smoke_check.py" "$ROOT"
