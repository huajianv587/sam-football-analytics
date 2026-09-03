#!/bin/bash
set -euo pipefail

JOB_ID="${1:?usage: backend/scripts/live_tunnel.sh <slurm-job-id>}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
set -a
# shellcheck disable=SC1090
source "$BACKEND_DIR/.env"
set +a

TC2_PORT="${TC2_PORT:-22}"
LIVE_PORT="${LIVE_PORT:-8010}"
TARGET="$TC2_USER@$TC2_HOST"
COMPUTE_NODE="$(ssh -p "$TC2_PORT" "$TARGET" "squeue -h -j '$JOB_ID' -o %N")"
if [[ -z "$COMPUTE_NODE" || "$COMPUTE_NODE" == "(null)" ]]; then
  echo "Job $JOB_ID is not running yet. Check squeue and retry."
  exit 1
fi

echo "Forwarding localhost:$LIVE_PORT to $COMPUTE_NODE:$LIVE_PORT"
exec ssh -p "$TC2_PORT" -N -L "$LIVE_PORT:$COMPUTE_NODE:$LIVE_PORT" "$TARGET"
