#!/usr/bin/env bash
# Work out which MIG slice each model should run on, and print the exact
# launch commands.
#
# Gemma (4B LLM) and Svara (3B autoregressive TTS) have been sharing one MIG
# slice and starving each other: measured alone, Gemma does 34 tok/s and
# Svara does a clause in 0.96s; during a live call those become ~3 tok/s and
# 2.6-7.8s. Putting them on separate slices removes that.
#
# The failure mode this guards against is silent. CUDA_VISIBLE_DEVICES must
# be set for BOTH processes -- pin only one and the other still sees every
# device, lands on the same slice, and the contention looks unchanged.
#
#   bash setup_mig_split.sh

echo "=============================================================="
echo " MIG DEVICES VISIBLE TO THIS CONTAINER"
echo "=============================================================="

raw=$(nvidia-smi -L 2>/dev/null)
if [ -z "$raw" ]; then
    echo "  nvidia-smi -L returned nothing - cannot see any GPU."
    exit 1
fi
echo "$raw" | sed 's/^/  /'
echo

# Pull out "<memory-gb> <profile> <uuid>" per MIG device, biggest first.
parsed=$(echo "$raw" | grep 'MIG-' | while IFS= read -r line; do
    profile=$(printf '%s' "$line" | grep -oE '[0-9]+g\.[0-9]+gb' | head -1)
    uuid=$(printf '%s' "$line" | sed -n 's/.*UUID: \(MIG-[^)]*\)).*/\1/p')
    gb=$(printf '%s' "$profile" | sed -n 's/.*\.\([0-9]*\)gb/\1/p')
    [ -n "$uuid" ] && echo "${gb:-0} ${profile:-unknown} ${uuid}"
done | sort -rn)

count=$(printf '%s\n' "$parsed" | grep -c . )

if [ "$count" -lt 2 ]; then
    echo "=============================================================="
    echo " PROBLEM: only $count MIG device visible"
    echo "=============================================================="
    cat <<'EOF'
  The new slice exists on the host but this container cannot see it.
  Creating a MIG slice does not automatically expose it to a running
  container -- the pod/container has to request it, which usually needs a
  restart or an admin change.

  Until two devices appear here, splitting the models is not possible and
  nothing below will help.
EOF
    exit 1
fi

big=$(printf '%s\n' "$parsed" | sed -n '1p')
small=$(printf '%s\n' "$parsed" | sed -n '2p')

big_profile=$(echo "$big" | awk '{print $2}')
big_uuid=$(echo "$big" | awk '{print $3}')
small_profile=$(echo "$small" | awk '{print $2}')
small_uuid=$(echo "$small" | awk '{print $3}')

echo "=============================================================="
echo " ASSIGNMENT"
echo "=============================================================="
printf "  Gemma (4B LLM)   -> %-12s %s\n" "$big_profile" "$big_uuid"
printf "  Svara (3B TTS)   -> %-12s %s\n" "$small_profile" "$small_uuid"
echo
echo "  Gemma gets the larger slice: it is the bigger model and the"
echo "  dominant latency cost (~9s of a ~11.6s turn)."
echo

cat <<EOF
==============================================================
 TERMINAL A - SVARA (TTS)
==============================================================
pkill -f "api/server.py"
sleep 2
conda activate svara
cd ~/newollama-volume/svara-tts-inference
export CUDA_VISIBLE_DEVICES=$small_uuid
python api/server.py

==============================================================
 TERMINAL B - GEMMA (LLM)
==============================================================
pkill -f gemma_server.py
sleep 2
conda activate gemma4-agent
cd ~/newollama-volume/gemma4-agent
export CUDA_VISIBLE_DEVICES=$big_uuid
taskset -c 0-15 python gemma_server.py

==============================================================
 THEN VERIFY
==============================================================
bash check_services.sh

  The GPU / MIG ASSIGNMENT section must show two DIFFERENT uuids.
  If either says "(unset - sees ALL visible devices)" the pin did not
  take and the models are still sharing a slice.

==============================================================
 IF SVARA FAILS WITH CUDA OUT OF MEMORY
==============================================================
  vLLM grabs ~90% of GPU memory for its KV cache by default. That was
  fine on ${big_profile%gb}gb but is tight on ${small_profile%gb}gb.
  Look for gpu_memory_utilization in:
      ~/newollama-volume/svara-tts-inference/.env
      ~/newollama-volume/svara-tts-inference/api/server.py
  and lower it (try 0.7). Single-stream TTS needs very little KV cache.
EOF
