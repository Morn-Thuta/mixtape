#!/usr/bin/env bash
# One-time RAPIDS environment setup inside WSL2 Ubuntu (no sudo needed).
# Safe to re-run: downloads are cached by uv.
set -euo pipefail
REPO="/mnt/e/Nvidia Projects/mixtape"
VENV="$HOME/.venvs/mixtape"
export UV_HTTP_TIMEOUT=600          # big CUDA wheels; NVIDIA's index can be slow
export UV_CONCURRENT_DOWNLOADS=4

if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

uv venv --clear --python 3.12 "$VENV"
source "$VENV/bin/activate"
cd "$REPO"

# PyTorch must match the Windows NVIDIA driver (566.x supports CUDA <= 12.7),
# so pin the CUDA 12.6 build instead of the default CUDA 13 one.
uv pip install torch --torch-backend=cu126
# RAPIDS (CUDA 12) + everything else
uv pip install -r requirements.txt -r requirements-gpu.txt \
  --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
uv pip install -e .

python -c "import cudf, cugraph, cupy, torch; print('cudf', cudf.__version__, '| cugraph', cugraph.__version__, '| torch', torch.__version__, 'cuda', torch.cuda.is_available()); print(cudf.Series([1,2,3]).sum())"
