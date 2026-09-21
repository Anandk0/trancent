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

# Always resolve to the directory containing this script, regardless of where
# the caller ran it from. The old approach (cd dirname && zip DIR) made DIR
# become the whole parent volume when called from ~.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$ROOT/divya-code.zip}"

# Use an absolute path for DEST so it is findable after we cd into ROOT.
DEST="$(cd "$(dirname "$DEST")" && pwd)/$(basename "$DEST")"

echo "Packing: $ROOT"
echo "Output:  $DEST"
echo

# Zip from inside the trancent dir using '.', so the archive always contains
# paths like  trancent/agent_api.py  regardless of how the script was invoked.
cd "$ROOT"
zip -r "$DEST" . \
    --exclude "./.git/*" \
    --exclude "./__pycache__/*" \
    --exclude "./*/__pycache__/*" \
    --exclude "./*/*/__pycache__/*" \
    --exclude "./*.pyc" \
    --exclude "./*/*.pyc" \
    --exclude "./*.egg-info/*" \
    --exclude "./logs/*" \
    --exclude "./parler/*" \
    --exclude "./.cache/*" \
    --exclude "./divya-code.zip"

SIZE=$(du -sh "$DEST" | awk '{print $1}')
echo "Done: $DEST  ($SIZE)"
echo
echo "To unpack on the new server:"
echo "  unzip divya-code.zip -d trancent"
echo "  cd trancent"
echo "  bash install.sh"
