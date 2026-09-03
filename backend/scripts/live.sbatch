#!/bin/bash
#SBATCH --job-name=pitchvision-live
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-live-%j.out
#SBATCH --error=slurm-live-%j.err

set -euo pipefail

SESSION_DIR="$1"
CODE_DIR="${2:-$(dirname "$SESSION_DIR")/code}"
REMOTE_ROOT="$(dirname "$(dirname "$SESSION_DIR")")"
RUNTIME_DIR="${SAM2_RUNTIME_DIR:-$REMOTE_ROOT/runtime}"
LIVE_PORT="${LIVE_PORT:-8010}"

mkdir -p "$SESSION_DIR" "$RUNTIME_DIR/cache/ultralytics"
exec >"$SESSION_DIR/live.log" 2>&1

module load anaconda/25.5.1 cuda/12.8.0
export PYTHONPATH="$CODE_DIR:$RUNTIME_DIR/sam2"
export LIVE_SEG_MODEL="${LIVE_SEG_MODEL:-$RUNTIME_DIR/checkpoints/yolo11s-seg.pt}"
export LIVE_SAM_CHECKPOINT="${LIVE_SAM_CHECKPOINT:-$RUNTIME_DIR/checkpoints/sam2.1_hiera_base_plus.pt}"
export LIVE_SAM_CONFIG="${LIVE_SAM_CONFIG:-configs/sam2.1/sam2.1_hiera_b+.yaml}"
export LIVE_SAM_ENABLED="${LIVE_SAM_ENABLED:-true}"
export LIVE_IMAGE_SIZE="${LIVE_IMAGE_SIZE:-640}"
export LIVE_CONFIDENCE="${LIVE_CONFIDENCE:-0.25}"
export LIVE_DEVICE="${LIVE_DEVICE:-0}"
export YOLO_CONFIG_DIR="$RUNTIME_DIR/cache/ultralytics"
export TORCH_HOME="$RUNTIME_DIR/torch-cache"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1

hostname > "$SESSION_DIR/compute-node.txt"
echo "$LIVE_PORT" > "$SESSION_DIR/port.txt"
echo "Starting generic live person segmentation on $(hostname):$LIVE_PORT"
conda run --no-capture-output -p "$RUNTIME_DIR/env" \
  uvicorn live.server:app --host 0.0.0.0 --port "$LIVE_PORT"
