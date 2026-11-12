#!/bin/bash
# 5-minute check that a GPU job works before submitting real runs: Python, torch + CUDA, visible GPUs, offline
# DistilBERT. Submit from ~/NLProfessionals on the cluster:  sbatch cluster/gpu_test.sbatch
#SBATCH --job-name=gpu-test
#SBATCH --partition=gpu-2xA100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
echo "node $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
uv run python - <<'EOF'
import sys, time, torch, yaml
from transformers import AutoModelForSequenceClassification
print("python", sys.executable, sys.version.split()[0])
print("torch", torch.__version__, "| cuda available", torch.cuda.is_available(), "| visible GPUs", torch.cuda.device_count())
print("device", torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize(); t = time.time()
for _ in range(20):
    y = x @ x
torch.cuda.synchronize(); print(f"20 matmuls 4096^2: {time.time() - t:.3f} s")
name = yaml.safe_load(open("configs/distilbert.yaml"))["pretrained"]
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).cuda()
print("DistilBERT loaded offline:", sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
EOF
echo "GPU TEST OK"
