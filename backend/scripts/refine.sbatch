#!/bin/bash
#SBATCH --job-name=sam-refine
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

JOB_DIR="$1"
REMOTE_ROOT="$(dirname "$(dirname "$(dirname "$JOB_DIR")")")"
RUNTIME_DIR="${SAM2_RUNTIME_DIR:-$REMOTE_ROOT/runtime}"
exec >"$JOB_DIR/job.log" 2>&1

module load anaconda/25.5.1 cuda/12.8.0
export SAM2_CHECKPOINT="$RUNTIME_DIR/checkpoints/sam2.1_hiera_large.pt"
export SAM2_MODEL_CONFIG="configs/sam2.1/sam2.1_hiera_l.yaml"
export SAM_MODEL_TIER="large"
export PYTHONPATH="$JOB_DIR:$RUNTIME_DIR/sam2"
export TORCH_HOME="$RUNTIME_DIR/torch-cache"
export TRITON_CACHE_DIR="$RUNTIME_DIR/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$RUNTIME_DIR/torchinductor-cache"
export PYTHONUNBUFFERED=1
export SAM_OBJECT_BATCH=1
export SAM_MIN_DETECTION_IOU=0.1
export SAM_WINDOW_FRAMES=90
export SAM_WINDOW_OVERLAP=15
export SAM_VOS_OPTIMIZED=false
export SAM_BIDIRECTIONAL=true
export SAM_PROMPTS_PER_TRACK=3

conda run --no-capture-output -p "$RUNTIME_DIR/env" python -u -m worker.refine "$JOB_DIR"
