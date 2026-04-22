#!/usr/bin/env bash
# Boots the PysaacRC FastAPI server.
# Binds to 0.0.0.0 so WSL2's localhost-forwarding lets the Windows-side
# Cloudflare tunnel reach it at http://localhost:8787.
# Cloudflare is managed on the Windows host — this script only starts uvicorn.

set -euo pipefail

# Ensure we're in the correct Python environment
if [ -f "$HOME/venv/bin/activate" ]; then
    source "$HOME/venv/bin/activate"
fi

export PYSAAC_DATA="${PYSAAC_DATA:-$HOME/.pysaac}"
mkdir -p "$PYSAAC_DATA"

cd "$(dirname "$0")/../.."  # repo parent, so `-m PySaacSim.server.app` resolves

nohup uvicorn PySaacSim.server.app:app \
    --host 0.0.0.0 \
    --port 8787 \
    --proxy-headers \
    --forwarded-allow-ips='*' \
    --reload > "$PYSAAC_DATA/server.log" 2>&1 &

echo "PysaacRC server started in the background (PID $!). Logs loosely tailed in $PYSAAC_DATA/server.log"
