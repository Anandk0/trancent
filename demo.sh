#!/usr/bin/env bash
# Walk the pipeline one service at a time, showing the request and the
# response for each.
#
# Each step pauses first, so the request can be read out before it runs and
# the response can be talked through before the next one scrolls it away.
#
#   bash demo.sh

PAUSE=${PAUSE:-1}          # PAUSE=0 to run straight through

hr()   { printf '\n\033[1m%s\033[0m\n' "=============================================================="; }
step() { hr; printf '\033[1m %s\033[0m\n' "$1"; printf '%s\n\n' "=============================================================="; }
say()  { printf '\033[2m%s\033[0m\n' "$1"; }
wait_key() { [ "$PAUSE" = "1" ] && { printf '\n\033[2m   [enter to continue]\033[0m'; read -r _; }; }

jsonfmt() { python3 -c "
import json,sys
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except Exception:
    print(raw[:600]); sys.exit()
# Audio payloads are hundreds of KB of base64 - show the size, not the blob.
for k in ('audio_b64',):
    if isinstance(d, dict) and k in d and isinstance(d[k], str):
        d[k] = f'<{len(d[k])//1024} KB of base64 WAV audio>'
print(json.dumps(d, ensure_ascii=False, indent=2)[:1400])
"; }

SESSION="demo-$(date +%H%M%S)"

# ---------------------------------------------------------------
step "0.  WHAT IS RUNNING"
# ---------------------------------------------------------------
say "Each stage is a separate process listening on its own port."
echo
printf "%-7s %-26s %s\n" "PORT" "FILE" "ROLE"
printf "%-7s %-26s %s\n" "----" "----" "----"
printf "%-7s %-26s %s\n" "8080" "call_server.py"      "web page + microphone"
printf "%-7s %-26s %s\n" "8002" "agent_api.py"        "orchestrator"
printf "%-7s %-26s %s\n" "8001" "asr_api.py"          "speech -> text"
printf "%-7s %-26s %s\n" "8000" "gemma_server.py"     "text -> reply"
printf "%-7s %-26s %s\n" "8003" "svara_tts_api.py"    "reply -> speech"
printf "%-7s %-26s %s\n" "8095" "api/server.py"       "TTS model (vLLM)"
echo
say "Live process list:"
for p in call_server.py agent_api.py asr_api.py gemma_server.py svara_tts_api.py "api/server.py"; do
    pids=$(pgrep -f "$p" | tr '\n' ' ')
    if [ -n "$pids" ]; then
        printf "   \033[32m●\033[0m %-22s pid %s\n" "$p" "$pids"
    else
        printf "   \033[31m○\033[0m %-22s NOT RUNNING\n" "$p"
    fi
done
wait_key

# ---------------------------------------------------------------
step "1.  SPEECH RECOGNITION        asr_api.py      :8001"
# ---------------------------------------------------------------
say "First we need audio. We synthesise a sentence, then feed that audio"
say "back into recognition - so both services are demonstrated at once."
echo
SEED="हमारा एमबीए प्रोग्राम दो साल का है"
say "Sentence to synthesise:  $SEED"
echo
curl -s -m 90 -X POST http://127.0.0.1:8003/synthesize \
     -H 'Content-Type: application/json' \
     -d "{\"text\": \"$SEED\"}" \
  | python3 -c "
import json,sys,base64
d=json.load(sys.stdin)
open('/tmp/demo_clip.wav','wb').write(base64.b64decode(d['audio_b64']))
print('  wrote /tmp/demo_clip.wav')
"
echo
say "REQUEST"
cat <<'REQ'
  curl -X POST http://127.0.0.1:8001/transcribe \
       -F 'file=@/tmp/demo_clip.wav' \
       -F 'language=hi' \
       -F 'decoding=ctc'
REQ
echo
say "RESPONSE"
curl -s -m 60 -X POST http://127.0.0.1:8001/transcribe \
     -F 'file=@/tmp/demo_clip.wav' \
     -F 'language=hi' \
     -F 'decoding=ctc' | jsonfmt
wait_key

# ---------------------------------------------------------------
step "2.  LANGUAGE MODEL           gemma_server.py  :8000"
# ---------------------------------------------------------------
say "Text in, Divya's reply out. The session id is what gives her memory -"
say "the server keeps this call's history under that key."
echo
say "REQUEST"
cat <<REQ
  curl -X POST http://127.0.0.1:8000/chat \\
       -H 'Content-Type: application/json' \\
       -d '{"session_id": "$SESSION",
            "text": "एमबीए में कौन कौन सी specialization हैं?"}'
REQ
echo
say "RESPONSE"
curl -s -m 120 -X POST http://127.0.0.1:8000/chat \
     -H 'Content-Type: application/json' \
     -d "{\"session_id\": \"$SESSION\", \"text\": \"एमबीए में कौन कौन सी specialization हैं?\"}" \
  | jsonfmt
wait_key

# ---------------------------------------------------------------
step "2b. PROVING THE MEMORY"
# ---------------------------------------------------------------
say "Same session id, and a follow-up that only makes sense if she"
say "remembers the previous question."
echo
say "REQUEST"
cat <<REQ
  curl -X POST http://127.0.0.1:8000/chat \\
       -H 'Content-Type: application/json' \\
       -d '{"session_id": "$SESSION",
            "text": "उनमें से सबसे popular कौन सा है?"}'
REQ
echo
say "RESPONSE"
curl -s -m 120 -X POST http://127.0.0.1:8000/chat \
     -H 'Content-Type: application/json' \
     -d "{\"session_id\": \"$SESSION\", \"text\": \"उनमें से सबसे popular कौन सा है?\"}" \
  | jsonfmt
echo
say "She answers about specialisations without being told the subject again."
wait_key

# ---------------------------------------------------------------
step "3.  SPEECH SYNTHESIS         svara_tts_api.py :8003"
# ---------------------------------------------------------------
say "Text in, Hindi audio out. The adapter picks the voice and forwards to"
say "the model server on 8095."
echo
say "REQUEST"
cat <<'REQ'
  curl -X POST http://127.0.0.1:8003/synthesize \
       -H 'Content-Type: application/json' \
       -d '{"text": "जी बिलकुल, मैं आपकी मदद करती हूँ।"}'
REQ
echo
say "RESPONSE"
curl -s -m 90 -X POST http://127.0.0.1:8003/synthesize \
     -H 'Content-Type: application/json' \
     -d '{"text": "जी बिलकुल, मैं आपकी मदद करती हूँ।"}' | jsonfmt
echo
say "The adapter also reports which backend it is forwarding to:"
curl -s -m 5 http://127.0.0.1:8003/health | jsonfmt
wait_key

# ---------------------------------------------------------------
step "4.  THE WHOLE PIPELINE       agent_api.py     :8002"
# ---------------------------------------------------------------
say "One call that runs all three. Passing 'transcript' skips recognition,"
say "so this is the language model and synthesis end to end."
echo
say "Watch the server terminal - the [TIMING] lines print live."
echo
say "REQUEST"
cat <<REQ
  curl -N -X POST http://127.0.0.1:8002/chat/stream \\
       -F 'transcript=प्लेसमेंट के बारे में बताइए' \\
       -F 'session_id=$SESSION' \\
       -F 'language=hi'
REQ
echo
say "RESPONSE  (streamed - one event per clause, as it becomes ready)"
echo
curl -s -N -m 180 -X POST http://127.0.0.1:8002/chat/stream \
     -F 'transcript=प्लेसमेंट के बारे में बताइए' \
     -F "session_id=$SESSION" \
     -F 'language=hi' \
  | python3 -u -c "
import json,sys,time
t0=time.time()
for line in sys.stdin:
    line=line.strip()
    if not line.startswith('data: '): continue
    try: e=json.loads(line[6:])
    except Exception: continue
    t=time.time()-t0
    k=e.get('type')
    if k=='audio_chunk':
        tag='filler' if e.get('is_filler') else 'reply '
        print(f'  [{t:6.2f}s] {tag} chunk {e[\"chunk_index\"]}  '
              f'{len(e.get(\"audio_b64\",\"\"))//1024:>4} KB   {e.get(\"sentence\",\"\")[:44]}')
    elif k=='asr_final':
        print(f'  [{t:6.2f}s] heard  : {e.get(\"transcript\",\"\")}')
    elif k=='gemma_start':
        print(f'  [{t:6.2f}s] model started generating')
    elif k=='done':
        ti=e.get('timings',{}) or {}
        print()
        for lbl,key in (('first clause ready','gemma_first_sentence_s'),
                        ('first clause spoken','tts_first_chunk_s'),
                        ('FIRST AUDIO','first_audio_ready_s'),
                        ('whole turn','agent_total_s')):
            if key in ti: print(f'    {lbl:<20} {ti[key]:>7.2f} s')
    elif k=='error':
        print('  ERROR:', e.get('error'))
"
hr
say "Done. Each stage answered on its own, and then all of them together."
echo
