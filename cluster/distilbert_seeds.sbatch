#!/bin/bash
# DistilBERT final runs on the Elja cluster (seeds come from configs/distilbert.yaml). Every GPU job gets a whole node
# (partition OverSubscribe=EXCLUSIVE: 2 A100s, billed for all 128 cores), so one job runs one encoding per GPU in
# parallel. Submit from ~/NLProfessionals on the cluster, at most as many encodings as GPUs:
#   sbatch --job-name=dbert-e1e2 cluster/distilbert_seeds.sbatch e1_ppfle e2_ppfle_bins
# Each encoding runs every unfinished DistilBERT final run (NetFlow Exp 2/3 + Edge-IIoTset). Finished runs are
# skipped, so resubmitting after a time-out continues where it stopped. Parallel runs share results/results.csv
# safely (src/eval/results.py locks it). The SLURM lines below are this job's settings (partitions: `sinfo`).
#SBATCH --job-name=distilbert
#SBATCH --partition=gpu-2xA100      # 2 A100 (40 GB) per node; max 2 days per job
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00           # shorter requests start sooner; a time-out only means: resubmit
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
(( $# >= 1 )) || { echo "usage: sbatch cluster/distilbert_seeds.sbatch <encoding> [<encoding> ...]"; exit 1; }
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh                 # uv's Python, no .venv changes, offline Hugging Face (see that file)
IFS=, read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"
(( $# <= ${#gpus[@]} )) || { echo "$# encodings but only ${#gpus[@]} GPUs"; exit 1; }
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / $# ))

echo "job $SLURM_JOB_ID on $(hostname), GPUs $CUDA_VISIBLE_DEVICES, encodings $*, started $(date)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
pids=()
for i in $(seq 0 $(( $# - 1 ))); do
    encoding="${@:i+1:1}"
    CUDA_VISIBLE_DEVICES="${gpus[i]}" uv run ftw distilbert_final distilbert_edge_final --encoding "$encoding" \
        > "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${encoding}.log" 2>&1 &
    pids+=($!)
    echo "GPU ${gpus[i]}: $encoding (log logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${encoding}.log)"
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "finished $(date), exit status $status"
exit $status
