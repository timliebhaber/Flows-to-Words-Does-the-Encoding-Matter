#!/bin/bash
# One-time setup of the Python environment on the Elja login node (~3-5 min, downloads ~6 GB). Rerun after a change
# to uv.lock. Usage, from ~/NLProfessionals on the cluster:  bash cluster/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/env.sh

# The A100 nodes run NVIDIA driver 535 (CUDA <= 12.2). The torch in uv.lock is built for CUDA 13 (driver >= 580), so
# the same torch version is reinstalled from PyTorch's CUDA 12.6 index (12.x builds run on driver 535). Laptop: unchanged.
TORCH_INDEX=https://download.pytorch.org/whl/cu126

uv python install 3.12
uv sync --locked
torch_version=$(uv run python -c "import importlib.metadata as m; print(m.version('torch').split('+')[0])")
uv pip install --python .venv/bin/python --index-url "$TORCH_INDEX" "torch==${torch_version}+${TORCH_INDEX##*/}"

# Pre-download the pretrained DistilBERT (name from the config) so jobs need no internet.
HF_HUB_OFFLINE=0 uv run python -c "
import yaml
from huggingface_hub import snapshot_download
name = yaml.safe_load(open('configs/distilbert.yaml'))['pretrained']
print('cached', snapshot_download(name, allow_patterns=['*.json', '*.txt', '*.safetensors']))"

uv run python -c "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda build', torch.version.cuda)"
