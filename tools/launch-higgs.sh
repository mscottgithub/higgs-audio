#!/usr/bin/env bash
set -euo pipefail

PY=/home/mintdude/venvs/higgs-py310/bin/python
APP=tools.higgs_local_server:app
WORKDIR=/home/mintdude/Github/sparky/higgs

# Nuke any inherited LD_LIBRARY_PATH (was pulling from voice-ai)
unset LD_LIBRARY_PATH
# Minimal, known-good library path
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"

# Local-only caches (safe for Higgs)
export HF_HOME=/mnt/data3/VoiceModels/huggingface
export HUGGINGFACE_HUB_CACHE=/mnt/data3/VoiceModels/huggingface
export TRANSFORMERS_CACHE=/mnt/data3/VoiceModels/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[HIGGS LAUNCH] python: $PY"
echo "[HIGGS LAUNCH] LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<unset>}"
echo "[HIGGS LAUNCH] HF_HOME: ${HF_HOME}"

cd "$WORKDIR"
exec "$PY" -m uvicorn "$APP" --host 0.0.0.0 --port 8010
