import time
import json
import base64
import uuid
import threading

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool


AGENT_STREAM_URL = "http://127.0.0.1:8002/chat/stream"
AGENT_BATCH_URL = "http://127.0.0.1:8002/chat"
ASR_URL = "http://127.0.0.1:8001/transcribe"
HOST = "0.0.0.0"
PORT = 8080

app = FastAPI(title="MBA Call Browser Interface")

# Persistent HTTP session for fast localhost forwarding
http_session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=15, pool_maxsize=30)
http_session.mount("http://", adapter)
http_session.mount("https://", adapter)


# ============================================================
# STREAMING ASR STATE
# ------------------------------------------------------------
# The browser POSTs the growing (cumulative) audio buffer to /asr_partial
# every ~700ms WHILE the caller is still speaking. Each partial is
# transcribed and the latest transcript is stored per session here.
#
# When the caller stops (VAD silence), /process pops the already-computed
# transcript and forwards it to the agent as `transcript=`, so the agent
# does NO ASR right before Gemma. The heavy ASR compute has already happened,
# overlapped with the caller's speech (dead time) instead of sitting in the
# critical path where it stalls Gemma's first token.
# ============================================================

LATEST_TRANSCRIPT = {}          # session_id -> {"text": str, "ts": float}
_transcript_lock = threading.Lock()


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok", "service": "call-server", "mode": "http-streaming-asr"}


# ============================================================
# BROWSER UI
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML_PAGE)


# ============================================================
# PARTIAL ASR ENDPOINT (called repeatedly DURING speech)
# Browser POSTs the cumulative audio buffer (valid WebM from the start).
# We transcribe it and cache the latest transcript for this session.
# ============================================================

@app.post("/asr_partial")
async def asr_partial(request: Request):
    audio_bytes = await request.body()
    session_id = request.headers.get("X-Session-Id", "")

    if not audio_bytes or not session_id:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": "missing audio or session id"}
        )

    def _run_asr():
        r = http_session.post(
            ASR_URL,
            files={"file": ("partial.webm", audio_bytes, "audio/webm")},
            data={"language": "hi", "decoding": "ctc"},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("text", "").strip()

    try:
        # Offload the blocking ASR call so the event loop stays free for the
        # SSE forwarding on /process.
        text = await run_in_threadpool(_run_asr)
    except Exception as e:
        print("ASR PARTIAL ERROR:", e)
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})

    if text:
        with _transcript_lock:
            LATEST_TRANSCRIPT[session_id] = {"text": text, "ts": time.time()}

    return {"status": "ok", "transcript": text}


# ============================================================
# PROCESS ENDPOINT (called ONCE at end-of-speech)
# If a streamed partial transcript exists for this session, forward it to
# the agent as `transcript=` (skips ASR entirely -> no burst before Gemma).
# Otherwise fall back to forwarding the raw audio (agent does ASR).
# ============================================================

@app.post("/process")
async def process(request: Request):
    t_request_start = time.perf_counter()

    print("\n" + "=" * 60)
    print("CALL SERVER: NEW STREAMING REQUEST")

    audio_bytes = await request.body()
    session_id = request.headers.get("X-Session-Id", str(uuid.uuid4()))
    print("Session:", session_id, "| audio bytes:", len(audio_bytes))

    # Prefer the transcript already computed during speech.
    with _transcript_lock:
        cached = LATEST_TRANSCRIPT.pop(session_id, None)
    streamed_transcript = cached["text"] if cached else None

    if streamed_transcript:
        print(f"[STREAM-ASR] using pre-computed transcript: '{streamed_transcript}'")
    elif not audio_bytes:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": "No audio and no streamed transcript"}
        )
    else:
        print("[STREAM-ASR] no partial available; agent will run ASR on audio (fallback)")

    def stream_forwarder():
        try:
            if streamed_transcript:
                # Emit the asr_final event ourselves (the agent won't, since it
                # is skipping ASR), so the browser shows what the caller said.
                asr_evt = {
                    "type": "asr_final",
                    "transcript": streamed_transcript,
                    "asr_time_s": 0.0,
                    "streamed": True,
                }
                yield f"data: {json.dumps(asr_evt, ensure_ascii=False)}\n\n"
                post_kwargs = dict(
                    data={"session_id": session_id, "transcript": streamed_transcript}
                )
            else:
                post_kwargs = dict(
                    files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                    data={"session_id": session_id, "language": "hi"},
                )

            with http_session.post(
                AGENT_STREAM_URL,
                stream=True,
                timeout=180,
                **post_kwargs,
            ) as agent_resp:
                agent_resp.raise_for_status()
                for line in agent_resp.iter_lines(decode_unicode=True):
                    if line:
                        yield f"{line}\n\n"
        except Exception as e:
            print("AGENT STREAM FORWARD ERROR:", e)
            err_payload = json.dumps({"type": "error", "error": str(e)})
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(
        stream_forwarder(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================================
# BROWSER PAGE
# Single-page call interface with:
# - browser microphone via MediaRecorder
# - RMS-based voice activity detection
# - silence detection (~1.2 s)
# - PROGRESSIVE STREAMING ASR: cumulative audio POSTed every ~700ms while
#   the caller speaks, so the transcript is ready the instant they stop
# - in-flight concurrency lock (single request at a time)
# - HTTP streaming SSE consumption
# - gapless sequential Web Audio chunk queue player
# ============================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Jain MBA Admissions – Divya</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; background: #f9f9f9; }
  h1 { font-size: 1.4em; color: #333; }
  #status { margin: 16px 0; padding: 12px; border-radius: 8px; background: #eee; font-size: 0.95em; }
  #status.listening { background: #d4edda; color: #155724; }
  #status.processing { background: #fff3cd; color: #856404; }
  #status.speaking { background: #cce5ff; color: #004085; }
  #status.idle { background: #eee; color: #555; }
  button { padding: 12px 28px; font-size: 1em; border: none; border-radius: 8px; cursor: pointer; }
  #startBtn { background: #28a745; color: #fff; }
  #endBtn   { background: #dc3545; color: #fff; display: none; }
  #log { margin-top: 20px; font-size: 0.82em; color: #555; white-space: pre-wrap; max-height: 320px; overflow-y: auto; background: #fff; padding: 10px; border-radius: 6px; border: 1px solid #ddd; }
</style>
</head>
<body>
<h1>📞 Jain College MBA Admissions</h1>
<p>AI Admission Counselor – Divya (Streaming ASR)</p>

<button id="startBtn" onclick="startCall()">📞 Start Call</button>
<button id="endBtn"   onclick="endCall()">🔴 End Call</button>

<div id="status" class="idle">Press Start Call to begin.</div>
<div id="log"></div>

<script>
const SILENCE_MS       = 1200;   // ms of silence before sending
const RMS_THRESHOLD    = 0.012;  // voice activity threshold
const SAMPLE_RATE      = 16000;
const CHUNK_MS         = 100;    // analyser poll interval + recorder timeslice
const PARTIAL_MS       = 700;    // cadence of progressive ASR passes during speech

let mediaStream        = null;
let audioContext       = null;
let analyser           = null;
let mediaRecorder      = null;
let recordedChunks     = [];
let silenceTimer       = null;
let vadTimeoutId       = null;
let speaking           = false;
let callActive         = false;
let divyaSpeaking      = false;
let isProcessing       = false;  // client-side in-flight / processing lock
let sessionId          = null;

// Progressive streaming-ASR state
let partialTimer       = null;
let partialInFlight    = false;
let lastPartialText    = "";

// Sequential Audio Chunk Playback Queue
let audioQueue         = [];
let isPlayingQueue     = false;
let currentSource      = null;
let isStreamDone       = false;

function log(msg) {
  const el = document.getElementById("log");
  el.textContent += new Date().toLocaleTimeString() + "  " + msg + "\\n";
  el.scrollTop = el.scrollHeight;
}

function setStatus(text, cls) {
  const el = document.getElementById("status");
  el.textContent = text;
  el.className = cls;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function generateSessionId() {
  return "browser-" + Date.now() + "-" + Math.random().toString(36).slice(2, 8);
}

async function startCall() {
  sessionId = generateSessionId();
  log("Session: " + sessionId);

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch (e) {
    log("Microphone error: " + e);
    return;
  }

  audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }

  const source = audioContext.createMediaStreamSource(mediaStream);

  analyser = audioContext.createAnalyser();
  analyser.fftSize = 512;
  source.connect(analyser);

  callActive = true;
  isProcessing = false;
  divyaSpeaking = false;
  audioQueue = [];
  isPlayingQueue = false;
  isStreamDone = false;

  document.getElementById("startBtn").style.display = "none";
  document.getElementById("endBtn").style.display   = "inline-block";

  setStatus("🎙️ Listening...", "listening");
  log("Call started.");

  startListening();
}

function endCall() {
  callActive = false;
  isProcessing = false;
  divyaSpeaking = false;
  stopListening();
  stopPartialLoop();
  if (currentSource) {
    try { currentSource.stop(); } catch (e) {}
  }
  audioQueue = [];
  isPlayingQueue = false;

  if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
  if (audioContext) audioContext.close();
  document.getElementById("startBtn").style.display = "inline-block";
  document.getElementById("endBtn").style.display   = "none";
  setStatus("Call ended.", "idle");
  log("Call ended.");
}

function startListening() {
  if (!callActive || isProcessing || divyaSpeaking) return;

  recordedChunks = [];
  speaking       = false;
  lastPartialText = "";
  clearTimeout(silenceTimer);
  silenceTimer = null;
  clearTimeout(vadTimeoutId);
  vadTimeoutId = null;

  try {
    mediaRecorder = new MediaRecorder(mediaStream);
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
    mediaRecorder.onstop = onRecordingStop;
    mediaRecorder.start(CHUNK_MS);
  } catch (e) {
    log("MediaRecorder start error: " + e);
    return;
  }

  pollVAD();
  schedulePartial();
}

function stopListening() {
  clearTimeout(silenceTimer);
  silenceTimer = null;
  clearTimeout(vadTimeoutId);
  vadTimeoutId = null;
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
}

// ---- Progressive streaming ASR ----------------------------------------

function schedulePartial() {
  clearTimeout(partialTimer);
  if (!callActive || isProcessing || divyaSpeaking) return;
  partialTimer = setTimeout(runPartial, PARTIAL_MS);
}

function stopPartialLoop() {
  clearTimeout(partialTimer);
  partialTimer = null;
}

async function runPartial() {
  partialTimer = null;
  // Only transcribe while the caller is actively speaking this turn.
  if (!callActive || isProcessing || divyaSpeaking) return;
  if (!speaking || recordedChunks.length === 0 || partialInFlight) {
    schedulePartial();
    return;
  }

  partialInFlight = true;
  try {
    // Cumulative blob: valid WebM from the first chunk, so ffmpeg on the ASR
    // server can decode it standalone every time.
    const blob = new Blob(recordedChunks, { type: "audio/webm" });
    const buf = await blob.arrayBuffer();
    const resp = await fetch("asr_partial", {
      method: "POST",
      headers: { "Content-Type": "audio/webm", "X-Session-Id": sessionId },
      body: buf
    });
    if (resp.ok) {
      const j = await resp.json();
      if (j.transcript) {
        lastPartialText = j.transcript;
        setStatus("🎙️ Listening... (" + lastPartialText + ")", "listening");
      }
    }
  } catch (e) {
    // Partial failures are non-fatal; the /process fallback still works.
    console.warn("partial ASR error", e);
  }
  partialInFlight = false;
  schedulePartial();
}

// -----------------------------------------------------------------------

function pollVAD() {
  if (!callActive || isProcessing || divyaSpeaking) return;

  const buf = new Float32Array(analyser.fftSize);
  analyser.getFloatTimeDomainData(buf);

  let sum = 0;
  for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
  const rms = Math.sqrt(sum / buf.length);

  if (rms > RMS_THRESHOLD) {
    if (!speaking) {
      speaking = true;
      clearTimeout(silenceTimer);
      log("Voice detected (RMS " + rms.toFixed(4) + ")");
    } else {
      clearTimeout(silenceTimer);
    }
    silenceTimer = setTimeout(onSilence, SILENCE_MS);
  }

  if (callActive && !isProcessing && !divyaSpeaking) {
    vadTimeoutId = setTimeout(pollVAD, CHUNK_MS);
  }
}

function onSilence() {
  if (!speaking || !callActive || isProcessing || divyaSpeaking) return;

  // Engage processing lock immediately to block any duplicate requests
  isProcessing = true;
  isStreamDone = false;
  audioQueue = [];
  isPlayingQueue = false;

  stopPartialLoop();
  log("Silence detected – finalizing...");
  stopListening();
}

function onRecordingStop() {
  if (!speaking || recordedChunks.length === 0) {
    isProcessing = false;
    if (callActive && !divyaSpeaking) startListening();
    return;
  }

  const blob = new Blob(recordedChunks, { type: "audio/webm" });
  log("Audio blob: " + blob.size + " bytes"
      + (lastPartialText ? " | partial ASR ready" : " | no partial (fallback)"));
  sendAudioStream(blob);
}

async function sendAudioStream(blob) {
  isProcessing = true;
  setStatus("⏳ Processing...", "processing");

  // Let any in-flight partial finish so its transcript is cached server-side
  // before /process pops it. (Partial ASR runs during dead time, not before
  // Gemma, so this wait does not add to the critical path.)
  let waited = 0;
  while (partialInFlight && waited < 3000) {
    await sleep(50);
    waited += 50;
  }

  const t0 = performance.now();

  try {
    const arrayBuf = await blob.arrayBuffer();
    const response = await fetch("process", {
      method: "POST",
      headers: {
        "Content-Type": "audio/webm",
        "X-Session-Id": sessionId
      },
      body: arrayBuf
    });

    const t1 = performance.now();
    log("[TIMING] BROWSER_POST_ROUNDTRIP: " + ((t1 - t0) / 1000).toFixed(3) + "s");

    if (!response.ok) {
      log("Server error: " + response.status);
      resetListening();
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\\n\\n");
      buffer = lines.pop(); // keep remainder

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const jsonStr = line.slice(6).trim();
          if (jsonStr) {
            try {
              const event = JSON.parse(jsonStr);
              handleStreamEvent(event);
            } catch (err) {
              console.error("JSON parse error on SSE:", err, jsonStr);
            }
          }
        }
      }
    }

  } catch (e) {
    log("Fetch error: " + e);
    resetListening();
  }
}

function handleStreamEvent(event) {
  if (event.type === "asr_final") {
    log("You: " + event.transcript + (event.streamed ? "  [streamed ASR]" : ""));
    log("[TIMING] ASR: " + event.asr_time_s + "s");
  }
  else if (event.type === "gemma_final") {
    log("Divya: " + event.reply);
    log("[TIMING] GEMMA_TOTAL: " + event.gemma_time_s + "s");
  }
  else if (event.type === "audio_chunk") {
    log("Received Audio Chunk " + (event.chunk_index + 1) + "/" + event.total_chunks +
        " (First Audio: " + event.first_audio_latency_s + "s, TTS: " + event.tts_time_s + "s)");
    enqueueAudioChunk(event.audio_b64, event.sample_rate || 22050, event.first_audio_latency_s);
  }
  else if (event.type === "done") {
    isStreamDone = true;
    if (event.timings) {
      log("[TIMING] GEMMA_FIRST_SENTENCE: " + event.timings.gemma_first_sentence_s + "s");
      log("[TIMING] TTS_FIRST_CHUNK: " + event.timings.tts_first_chunk_s + "s");
      log("[TIMING] FIRST_AUDIO_READY: " + event.timings.first_audio_ready_s + "s");
      log("[TIMING] AGENT_TOTAL: " + event.timings.agent_total_s + "s");
    }
    // If no audio chunks were generated (e.g. empty reply), resume listening
    if (!isPlayingQueue && audioQueue.length === 0) {
      resetListening();
    }
  }
  else if (event.type === "error") {
    log("Stream stage error (" + event.stage + "): " + event.error);
    resetListening();
  }
}

async function enqueueAudioChunk(b64, sampleRate, firstAudioLatency) {
  if (!b64) return;

  try {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

    const t_decode_start = performance.now();
    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }
    const decoded = await audioContext.decodeAudioData(bytes.buffer);
    const t_decode_elapsed = (performance.now() - t_decode_start) / 1000;
    log("[TIMING] BROWSER_AUDIO_DECODE: " + t_decode_elapsed.toFixed(3) + "s");

    audioQueue.push(decoded);

    if (!isPlayingQueue) {
      playNextQueueChunk();
    }
  } catch (err) {
    console.error("Audio decode error:", err);
    if (!isPlayingQueue && isStreamDone && audioQueue.length === 0) {
      resetListening();
    }
  }
}

function playNextQueueChunk() {
  if (audioQueue.length === 0) {
    if (isStreamDone) {
      isPlayingQueue = false;
      divyaSpeaking = false;
      isProcessing = false;
      if (callActive) {
        setStatus("🎙️ Listening...", "listening");
        log("Response received. Resuming listening...");
        startListening();
      }
    } else {
      // Waiting for subsequent sentence chunks to arrive over SSE stream
      isPlayingQueue = false;
    }
    return;
  }

  isPlayingQueue = true;
  divyaSpeaking = true;
  setStatus("🔊 Divya is speaking...", "speaking");

  const buffer = audioQueue.shift();
  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(audioContext.destination);
  currentSource = source;

  const t_play_start = performance.now();
  source.onended = () => {
    const t_play_end = performance.now();
    log("[TIMING] BROWSER_AUDIO_PLAY: " + ((t_play_end - t_play_start) / 1000).toFixed(3) + "s");
    playNextQueueChunk();
  };

  source.start();
}

function resetListening() {
  isProcessing = false;
  divyaSpeaking = false;
  isPlayingQueue = false;
  isStreamDone = true;
  audioQueue = [];
  if (callActive) {
    setStatus("🎙️ Listening...", "listening");
    log("Response received. Resuming listening...");
    startListening();
  }
}
</script>
</body>
</html>
"""


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
