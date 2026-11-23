#!/usr/bin/env bash
set -euo pipefail
RADXA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_PYTHON="${DEPLOY_PYTHON:-$RADXA_ROOT/.venv/bin/python}"
if [[ ! -x "$DEPLOY_PYTHON" ]]; then
  echo 'Run bash setup.sh first, or set DEPLOY_PYTHON to a Python environment with requirements.txt installed.' >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "$DEPLOY_PYTHON" -u "$RADXA_ROOT/run.py" "$@"
