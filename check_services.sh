#!/usr/bin/env bash
# Show the true state of every service in the voice pipeline.
#
# Restarts have repeatedly half-failed in ways that look like application
# bugs: a dead Svara backend reads as "Divya can't hear me", a duplicate
# process holding a port means the restart silently kept running old code.
# This prints what is actually listening, and what is actually running, so
# those are visible immediately instead of being debugged as latency.
#
#   bash check_services.sh

echo "=============================================================="
echo " PORTS"
echo "=============================================================="
printf "%-6s %-22s %s\n" "PORT" "SERVICE" "STATUS"

check() {   # port, name, path
    local port=$1 name=$2 path=$3
    local code
    code=$(curl -s -o /dev/null -m 3 -w "%{http_code}" "http://127.0.0.1:${port}${path}" 2>/dev/null)
    local status
    case "$code" in
        000) status="DOWN  (nothing listening)" ;;
        200) status="ok    ($code)" ;;
        404) status="up    ($code - no route at ${path}, fine)" ;;
        *)   status="up?   ($code)" ;;
    esac
    printf "%-6s %-22s %s\n" "$port" "$name" "$status"
}

check 8000 "gemma_server"      "/health"
check 8001 "asr_api"           "/health"
check 8002 "agent_api"         "/health"
check 8003 "tts adapter"       "/health"
check 8080 "call_server"       "/health"
check 8095 "svara model"       "/health"
check 8007 "svara (alt port)"  "/health"

echo
echo "=============================================================="
echo " PROCESSES  (more than one of anything is a problem)"
echo "=============================================================="
for p in gemma_server.py asr_api.py agent_api.py svara_tts_api.py \
         kokoro_server.py call_server.py "api/server.py"; do
    n=$(pgrep -f "$p" | wc -l)
    if [ "$n" -eq 0 ]; then
        printf "  %-20s not running\n" "$p"
    elif [ "$n" -eq 1 ]; then
        printf "  %-20s running (pid %s)\n" "$p" "$(pgrep -f "$p" | tr '\n' ' ')"
    else
        printf "  %-20s *** %s COPIES *** pids: %s\n" "$p" "$n" "$(pgrep -f "$p" | tr '\n' ' ')"
    fi
done

echo
echo "=============================================================="
echo " EXPECTED"
echo "=============================================================="
cat <<'EOF'
  8000 gemma_server   up (404 on /health is normal - no such route)
  8001 asr_api        ok
  8002 agent_api      ok
  8003 tts adapter    ok   <- agent_api calls this; if DOWN there is no audio
  8080 call_server    ok
  8095 svara model    ok   <- the adapter forwards here; if DOWN there is no audio

  Exactly ONE copy of each process. Two copies means a restart failed to
  bind and the OLD code is still serving.
EOF
