#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
PORT="${LOCAL_LIVE_PORT:-8011}"

cd "$BACKEND_DIR"
export LIVE_DEVICE="${LIVE_DEVICE:-auto}"
export LIVE_SAM_ENABLED="${LIVE_SAM_ENABLED:-false}"
export LIVE_SEG_MODEL="${LIVE_SEG_MODEL:-yolo11m-seg.pt}"
exec uv run uvicorn live.server:app --host 127.0.0.1 --port "$PORT"
