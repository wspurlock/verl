#!/usr/bin/env bash
# Prepare a self-contained VAPO environment whose large files live under /workspace.

set -euo pipefail

WORKSPACE_ROOT=${WORKSPACE_ROOT:-/workspace}
VAPO_HOME=${VAPO_HOME:-"$WORKSPACE_ROOT/vapo"}
ENV_FILE="$VAPO_HOME/env.sh"

if [[ ! -d "$WORKSPACE_ROOT" || ! -w "$WORKSPACE_ROOT" ]]; then
    echo "$WORKSPACE_ROOT must be an existing writable mount" >&2
    exit 1
fi

mkdir -p \
    "$VAPO_HOME/cache/huggingface" \
    "$VAPO_HOME/cache/torch" \
    "$VAPO_HOME/cache/uv" \
    "$VAPO_HOME/cache/xdg" \
    "$VAPO_HOME/checkpoints" \
    "$VAPO_HOME/data/gsm8k" \
    "$VAPO_HOME/logs" \
    "$VAPO_HOME/ray" \
    "$VAPO_HOME/tmp"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

UV_CACHE_DIR="$VAPO_HOME/cache/uv" uv venv --python 3.12 "$VAPO_HOME/.venv"

{
    printf 'export VAPO_HOME=%q\n' "$VAPO_HOME"
    printf 'export VIRTUAL_ENV=%q\n' "$VAPO_HOME/.venv"
    printf 'export PATH=%q:$PATH\n' "$VAPO_HOME/.venv/bin"
    printf 'export HF_HOME=%q\n' "$VAPO_HOME/cache/huggingface"
    printf 'export HUGGINGFACE_HUB_CACHE=%q\n' "$VAPO_HOME/cache/huggingface/hub"
    printf 'export TRANSFORMERS_CACHE=%q\n' "$VAPO_HOME/cache/huggingface/transformers"
    printf 'export HF_HUB_ENABLE_HF_TRANSFER=0\n'
    printf 'export TORCH_HOME=%q\n' "$VAPO_HOME/cache/torch"
    printf 'export UV_CACHE_DIR=%q\n' "$VAPO_HOME/cache/uv"
    printf 'export XDG_CACHE_HOME=%q\n' "$VAPO_HOME/cache/xdg"
    printf 'export RAY_TMPDIR=%q\n' "$VAPO_HOME/ray"
    printf 'export TMPDIR=%q\n' "$VAPO_HOME/tmp"
    printf 'export TRAIN_FILE=%q\n' "$VAPO_HOME/data/gsm8k/train.parquet"
    printf 'export VAL_FILE=%q\n' "$VAPO_HOME/data/gsm8k/test.parquet"
    printf 'export OUTPUT_DIR=%q\n' "$VAPO_HOME/checkpoints/vapo-smoke"
} >"$ENV_FILE"

# shellcheck disable=SC1090
source "$ENV_FILE"

# Install vLLM first so its compatible PyTorch build is selected. The v1
# trainer also depends on TransferQueue, which is pinned in requirements.txt
# rather than the editable package metadata.
uv pip install "vllm==0.11.0"
uv pip install -r requirements.txt
uv pip install "trl==0.27.0"
uv pip install "numpy==2.2.6"
uv pip install --upgrade "scipy>=1.13.0"
uv pip install "transformers==4.57.1"
uv pip install hf-transfer hf-xet
uv pip install \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
uv pip install --no-deps -e .

# Reassert the Numba-compatible NumPy version after all dependency resolution.
# Keep this final: later installs may otherwise upgrade NumPy transitively.
uv pip install "numpy==2.2.6"

python3 - <<'PY'
import numba
import numpy
import scipy
import vllm

assert numpy.__version__ == "2.2.6", numpy.__version__
print(f"Verified runtime versions: NumPy {numpy.__version__}, Numba {numba.__version__}, "
      f"SciPy {scipy.__version__}, vLLM {vllm.__version__}")
PY

python3 examples/data_preprocess/gsm8k.py \
    --local_save_dir "$VAPO_HOME/data/gsm8k"

python3 scripts/diagnose.py

echo "Setup complete. Before running VAPO:"
echo "  source $ENV_FILE"
