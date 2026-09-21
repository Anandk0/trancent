#!/usr/bin/env bash
# Create a single zip of all code for the 3 model services plus supporting
# scripts. Does NOT include model weights (those are downloaded on first run).
#
# Usage:
#   bash pack.sh                        # creates divya-code.zip here
#   bash pack.sh /some/other/path.zip   # write to a specific path
#
# What's in the zip:
#   trancent/               <- everything from this repo (code only)
#     agent_api.py          <- orchestrator
#     call_server.py        <- telephony / Exotel
#     gemma_server.py       <- Gemma-4 LLM server  (service 1)
#     asr-server/
#       asr_api.py          <- ASR server           (service 2)
#     svara_tts_api.py      <- Svara adapter
#     start_all.sh          <- one-command startup
#     setup_svara_pod.sh    <- pod setup helper
#     install.sh            <- NEW: installs envs + downloads models
#     ... (all other .py, .sh, .md files)
#
# What is excluded:
#   __pycache__/  *.pyc  .git/  logs/  *.egg-info/
#   parler/kokoro/ subdirs with their own big deps (not used in live calls)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$ROOT/divya-code.zip}"

echo "Packing code from $ROOT -> $DEST"

cd "$(dirname "$ROOT")"
DIR="$(basename "$ROOT")"

zip -r "$DEST" "$DIR" \
    --exclude "$DIR/.git/*" \
    --exclude "$DIR/__pycache__/*" \
    --exclude "$DIR/*/__pycache__/*" \
    --exclude "$DIR/*/*/__pycache__/*" \
    --exclude "$DIR/*.pyc" \
    --exclude "$DIR/*/*.pyc" \
    --exclude "$DIR/*.egg-info/*" \
    --exclude "$DIR/logs/*" \
    --exclude "$DIR/parler/*" \
    -x "*.pyc"

SIZE=$(du -sh "$DEST" | awk '{print $1}')
echo "Done: $DEST  ($SIZE)"
echo
echo "To unpack on the new server:"
echo "  unzip divya-code.zip"
echo "  cd trancent"
echo "  bash install.sh"
