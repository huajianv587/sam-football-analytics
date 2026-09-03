#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$BACKEND_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${TC2_HOST:?TC2_HOST is required}"
: "${TC2_USER:?TC2_USER is required}"
TC2_PORT="${TC2_PORT:-22}"
: "${TC2_REMOTE_ROOT:?TC2_REMOTE_ROOT is required}"

REMOTE_LIVE="$TC2_REMOTE_ROOT/live"
REMOTE_CODE="$REMOTE_LIVE/code"
REMOTE_SESSION="$REMOTE_LIVE/session"
TARGET="$TC2_USER@$TC2_HOST"

ssh -p "$TC2_PORT" "$TARGET" "mkdir -p '$REMOTE_CODE' '$REMOTE_SESSION'"
rsync -az --delete \
  --exclude '.env' --exclude '.venv' --exclude '__pycache__' --exclude '._*' \
  -e "ssh -p $TC2_PORT" "$BACKEND_DIR/" "$TARGET:$REMOTE_CODE/"
JOB_ID="$(ssh -p "$TC2_PORT" "$TARGET" \
  "cd '$REMOTE_CODE' && sbatch --parsable scripts/live.sbatch '$REMOTE_SESSION' '$REMOTE_CODE'")"

echo "PitchVision live job submitted: $JOB_ID"
echo "After it starts, run: backend/scripts/live_tunnel.sh $JOB_ID"
