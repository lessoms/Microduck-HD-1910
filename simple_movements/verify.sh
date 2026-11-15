#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SIMPLE_MOVEMENTS_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found. Run: bash $ROOT/setup.sh" >&2
  exit 1
fi
export PYTHONPATH="$ROOT/src:$ROOT/runtime:$ROOT/vendor${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.cache/matplotlib}"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
cd "$ROOT"
exec "$PYTHON_BIN" -u "$ROOT/runtime/torch_entry.py" "$ROOT/runtime/verify_package.py" "$@"
