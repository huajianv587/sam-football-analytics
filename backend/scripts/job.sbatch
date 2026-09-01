#!/bin/bash
#SBATCH --job-name=sam-football
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

JOB_DIR="$1"
RUNTIME_DIR="${SAM2_RUNTIME_DIR:-$(dirname "$(dirname "$JOB_DIR")")/runtime}"
exec >"$JOB_DIR/job.log" 2>&1

module load anaconda/25.5.1 cuda/12.8.0
export SAM2_CHECKPOINT="$RUNTIME_DIR/checkpoints/sam2.1_hiera_large.pt"
export SAM2_MODEL_CONFIG="configs/sam2.1/sam2.1_hiera_l.yaml"
export PYTHONPATH="$JOB_DIR:$RUNTIME_DIR/sam2"
export TORCH_HOME="$RUNTIME_DIR/torch-cache"
export EASYOCR_MODULE_PATH="$RUNTIME_DIR/easyocr"
export TRITON_CACHE_DIR="$RUNTIME_DIR/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$RUNTIME_DIR/torchinductor-cache"
export PYTHONUNBUFFERED=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

conda run --no-capture-output -p "$RUNTIME_DIR/env" python -u -m worker.main "$JOB_DIR"
