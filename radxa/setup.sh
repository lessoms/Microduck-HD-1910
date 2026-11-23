#!/usr/bin/env bash
set -euo pipefail
RADXA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"
"$BOOTSTRAP_PYTHON" -c 'import sys; assert sys.version_info >= (3,10), "Python 3.10+ required"'
"$BOOTSTRAP_PYTHON" "$RADXA_ROOT/download_model.py"
if [[ ! -f "$RADXA_ROOT/config.json" ]]; then
  cp "$RADXA_ROOT/config.example.json" "$RADXA_ROOT/config.json"
fi
if [[ "${1:-}" == "--prepare-only" ]]; then
  echo 'Model verified and local configuration prepared; dependencies were not installed.'
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo 'Usage: bash setup.sh [--prepare-only]' >&2
  exit 2
fi
if [[ ! -x "$RADXA_ROOT/.venv/bin/python" ]]; then
  if ! "$BOOTSTRAP_PYTHON" -m venv "$RADXA_ROOT/.venv"; then
    echo 'Install the venv package for your Python (Debian: sudo apt install python3-venv), then rerun.' >&2
    exit 1
  fi
fi
"$RADXA_ROOT/.venv/bin/python" -m pip install --upgrade pip
"$RADXA_ROOT/.venv/bin/python" -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6,<3'
"$RADXA_ROOT/.venv/bin/python" -m pip install --only-binary=:all: 'numpy>=1.26,<3' 'aiohttp>=3.10,<4' 'pyserial>=3.5,<4'
"$RADXA_ROOT/.venv/bin/python" "$RADXA_ROOT/test_deployment.py"
echo 'Ready. Software-only test: bash run.sh --offline --seconds 3'
echo 'No servo torque or motion has been enabled.'
