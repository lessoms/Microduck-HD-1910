#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIMPLE="$ROOT/../simple_movements"
PYTHON_BIN="${SIMPLE_MOVEMENTS_PYTHON:-$SIMPLE/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found. Run: (cd $SIMPLE && uv sync --frozen)" >&2
  exit 1
fi
export PYTHONPATH="$ROOT/src:$ROOT/runtime:$SIMPLE/runtime:$SIMPLE/vendor${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SIMPLE/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON_BIN" -u "$SIMPLE/runtime/torch_entry.py" "$ROOT/runtime/combined.py" "$@"
