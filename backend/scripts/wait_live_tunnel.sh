#!/bin/bash
set -euo pipefail

JOB_ID="${1:?usage: wait_live_tunnel.sh <slurm-job-id>}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
set -a
source "$BACKEND_DIR/.env"
set +a

LIVE_PORT="${LIVE_PORT:-8010}"
TARGET="$TC2_USER@$TC2_HOST"
TUNNEL_PID=""

cleanup() {
  if [[ -n "$TUNNEL_PID" ]]; then
    kill "$TUNNEL_PID" 2>/dev/null || true
    wait "$TUNNEL_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT HUP INT TERM

while true; do
  NODE="$(ssh -p "${TC2_PORT:-22}" "$TARGET" "squeue -h -j '$JOB_ID' -o %N")"
  case "$NODE" in
    ""|"(null)") sleep 20 ;;
    *) ssh -p "${TC2_PORT:-22}" -N -o ExitOnForwardFailure=yes \
      -L "$LIVE_PORT:$NODE:$LIVE_PORT" "$TARGET" &
      TUNNEL_PID="$!"
      wait "$TUNNEL_PID" || true
      TUNNEL_PID=""
      sleep 2 ;;
  esac
done
