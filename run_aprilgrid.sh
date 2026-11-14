#!/usr/bin/env bash
set -euo pipefail
EGO2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$EGO2_ROOT"
exec "${EGO2_PYTHON:-python3}" -m ego2.aprilgrid "$@"
