#!/usr/bin/env bash
# Bring the whole voice pipeline up on one box, in dependency order.
#
# Restarts have repeatedly half-failed in ways that only show up as
# "Divya can't be heard": the Svara model server dies while the adapter in
# front of it stays up and forwards into nothing, or a duplicate process
# keeps the old code bound to the port. This starts each service only after
# the one it depends on answers, so a failure stops here instead of being
# debugged later as latency.
#
#   bash start_all.sh           # start everything
#   bash start_all.sh --stop    # stop everything
#
# Override a conda env if yours is named differently:
#   SVARA_ENV=svara GEMMA_ENV=gemma4-agent APP_ENV=gemma4-agent bash start_all.sh

set -u

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
LOGDIR="${LOGDIR:-$ROOT/logs}"

SVARA_ENV="${SVARA_ENV:-svara}"
GEMMA_ENV="${GEMMA_ENV:-gemma4-agent}"
# base is Python 3.8; agent_api.py uses `list[str]` annotations and needs
# 3.9+, so the root services run under gemma4-agent (3.10).
APP_ENV="${APP_ENV:-gemma4-agent}"

# Per-service directories. Each service lives in its own subdirectory with
# its own conda env; only the adapter, agent and call server run from ROOT.
SVARA_MODEL_DIR="${SVARA_MODEL_DIR:-$ROOT/svara-tts-inference}"
GEMMA_DIR="${GEMMA_DIR:-$ROOT/gemma4-agent}"
ASR_DIR="${ASR_DIR:-$ROOT/asr-server}"

mkdir -p "$LOGDIR"

# ---------------------------------------------------------------- helpers

port_code() {   # port path -> http status, 000 when nothing is listening
    curl -s -o /dev/null -m 3 -w "%{http_code}" "http://127.0.0.1:$1$2" 2>/dev/null
}

is_up() {       # a route that 404s still proves the server is bound
    local code
    code=$(port_code "$1" "$2")
    [ "$code" != "000" ]
}

wait_for() {    # port path label timeout_s
    local port=$1 path=$2 label=$3 timeout=$4
    local waited=0
    printf "  waiting for %-16s " "$label"
    while [ "$waited" -lt "$timeout" ]; do
        if is_up "$port" "$path"; then
            echo "up after ${waited}s"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    echo "TIMED OUT after ${timeout}s"
    echo
    echo "  $label never came up. Last 25 lines of its log:"
    tail -25 "$LOGDIR/$label.log" 2>/dev/null | sed 's/^/    /'
    return 1
}

start() {       # label env dir command port path timeout
    local label=$1 env=$2 dir=$3 cmd=$4 port=$5 path=$6 timeout=$7

    if is_up "$port" "$path"; then
        echo "  $label already up on $port - leaving it alone"
        return 0
    fi

    if [ ! -d "$dir" ]; then
        echo "  FAIL $label: directory not found: $dir"
        return 1
    fi

    local script="${cmd%% *}"
    if [ ! -f "$dir/$script" ]; then
        echo "  FAIL $label: $script not found in $dir"
        echo "        locate it with:  find ~ -name '$script' -not -path '*/.git/*' 2>/dev/null"
        echo "        then set its *_DIR variable (see the top of this script)"
        return 1
    fi

    echo "  starting $label  (env=$env, port=$port)"
    (
        # conda.sh dereferences PS1, which is unset in a non-interactive
        # shell, so it aborts under `set -u` before activating anything.
        set +u
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$env" || exit 1
        cd "$dir" || exit 1
        exec python $cmd
    ) >> "$LOGDIR/$label.log" 2>&1 &

    wait_for "$port" "$path" "$label" "$timeout"
}

# ---------------------------------------------------------------- stop

stop_all() {
    echo "Stopping all services..."
    for p in call_server.py agent_api.py asr_api.py gemma_server.py \
             svara_tts_api.py kokoro_server.py "api/server.py"; do
        pids=$(pgrep -f "$p")
        if [ -n "$pids" ]; then
            echo "  killing $p ($pids)"
            pkill -9 -f "$p"
        fi
    done
    sleep 2
    echo "Done."
}

if [ "${1:-}" = "--stop" ]; then
    stop_all
    exit 0
fi

# ---------------------------------------------------------------- start

echo "=============================================================="
echo " STARTING VOICE PIPELINE"
echo "=============================================================="
echo "  root : $ROOT"
echo "  logs : $LOGDIR"
echo

# Kill duplicates first. A second copy of a service holds the port and keeps
# serving old code, so a restart looks successful and changes nothing.
echo "Clearing stale processes..."
for p in call_server.py agent_api.py svara_tts_api.py; do
    n=$(pgrep -f "$p" | wc -l)
    if [ "$n" -gt 1 ]; then
        echo "  $p had $n copies - killing all, will restart one"
        pkill -9 -f "$p"
    fi
done
echo

# Svara first: the adapter in front of it is useless until this answers, and
# vLLM takes minutes to load weights.
echo "1. Svara model server (vLLM, slow to load)"
start "svara-model" "$SVARA_ENV" "$SVARA_MODEL_DIR" \
      "api/server.py" 8095 "/health" 420 || exit 1
echo

echo "2. Svara adapter"
start "svara-adapter" "$APP_ENV" "$ROOT" \
      "svara_tts_api.py" 8003 "/health" 60 || exit 1
echo

echo "3. Gemma LLM"
start "gemma" "$GEMMA_ENV" "$GEMMA_DIR" \
      "gemma_server.py" 8000 "/health" 300 || exit 1
echo

echo "4. ASR"
start "asr" "$APP_ENV" "$ASR_DIR" \
      "asr_api.py" 8001 "/health" 300 || exit 1
echo

echo "5. Agent API"
start "agent-api" "$APP_ENV" "$ROOT" \
      "agent_api.py" 8002 "/health" 90 || exit 1
echo

echo "6. Call server"
start "call-server" "$APP_ENV" "$ROOT" \
      "call_server.py" 8080 "/health" 60 || exit 1
echo

echo "=============================================================="
echo " ALL UP"
echo "=============================================================="
bash "$ROOT/check_services.sh"
