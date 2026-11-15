#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
cd "$ROOT"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
uv sync --frozen
.venv/bin/python -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda); assert torch.cuda.is_available(), "CUDA GPU/driver required"'
echo 'Ready. Start: bash run.sh'
