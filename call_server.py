import time
import base64
import uuid

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse


AGENT_URL = "http://127.0.0.1:8002/chat"
HOST = "0.0.0.0"
PORT = 8080

app = FastAPI(title="MBA Call Browser Interface")

# Persistent HTTP session for fast localhost forwarding
http_session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20)
http_session.mount("http://", adapter)
http_session.mount("https://", adapter)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok", "service": "call-server"}


# ============================================================
# BROWSER UI
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML_PAGE)


# ============================================================
# PROCESS ENDPOINT
# Browser POSTs raw audio bytes here.
# Returns base64-encoded WAV audio from Divya.
# ============================================================

@app.post("/process")
async def process(request: Request):
    t_request_start = time.perf_counter()

    print("\n" + "=" * 60)
    print("CALL SERVER: NEW REQUEST")

    # 1. Read raw audio bytes from browser
    t_read_start = time.perf_counter()
    audio_bytes = await request.body()
    t_read_elapsed = time.perf_counter() - t_read_start
    print(f"[TIMING] BROWSER_AUDIO_READ: {t_read_elapsed:.3f}s ({len(audio_bytes)} bytes)")

    if not audio_bytes:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": "No audio received"}
        )

    session_id = request.headers.get("X-Session-Id", str(uuid.uuid4()))
    print("Session:", session_id)

    # 2. Forward audio to agent_api /chat
    t_agent_start = time.perf_counter()
    try:
        agent_response = http_session.post(
            AGENT_URL,
            files={
                "file": ("audio.wav", audio_bytes, "audio/wav")
            },
            data={
                "session_id": session_id,
                "language": "hi"
            },
            timeout=180
        )
        agent_response.raise_for_status()
        agent_result = agent_response.json()
    except Exception as e:
        print("AGENT ERROR:", e)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )

    t_agent_elapsed = time.perf_counter() - t_agent_start
    print(f"[TIMING] AGENT_ROUNDTRIP: {t_agent_elapsed:.3f}s")

    # 3. Get base64 audio (in-memory direct pass-through, or fallback to file)
    t_encode_start = time.perf_counter()
    audio_b64 = agent_result.get("audio_b64")

    if not audio_b64:
        audio_file = agent_result.get("audio_file")
        if audio_file:
            try:
                with open(audio_file, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                print("Audio file read error:", e)

    t_encode_elapsed = time.perf_counter() - t_encode_start
    t_total = time.perf_counter() - t_request_start

    print(f"[TIMING] AUDIO_PREP: {t_encode_elapsed:.3f}s | TOTAL: {t_total:.3f}s")
    print("=" * 60)

    return JSONResponse(content={
        "status": "success",
        "session_id": session_id,
        "transcript": agent_result.get("transcript", ""),
        "reply": agent_result.get("reply", ""),
        "audio_b64": audio_b64,
        "sample_rate": agent_result.get("sample_rate", 22050),
        "timings": {
            "server_total_s": round(t_total, 3),
            "agent_roundtrip_s": round(t_agent_elapsed, 3)
        }
    })


# ============================================================
# BROWSER PAGE
# ============================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Jain MBA Admissions – Divya</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 640px; margin: 30px auto; padding: 24px; background: #f8fafc; color: #1e293b; }
  .card { background: #ffffff; border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.05); }
  h1 { font-size: 1.35em; margin-top: 0; color: #0f172a; }
  .subtitle { font-size: 0.9em; color: #64748b; margin-bottom: 20px; }
  #status { margin: 16px 0; padding: 14px 18px; border-radius: 8px; font-weight: 500; font-size: 0.95em; transition: all 0.2s ease; }
  #status.listening { background: #dcfce7; color: #15803d; border-left: 4px solid #22c55e; }
  #status.processing { background: #fef9c3; color: #a16207; border-left: 4px solid #eab308; }
  #status.speaking { background: #dbeafe; color: #1d4ed8; border-left: 4px solid #3b82f6; }
  #status.idle { background: #f1f5f9; color: #64748b; }
  .btn-group { display: flex; gap: 12px; margin-top: 16px; }
  button { padding: 12px 24px; font-size: 1em; font-weight: 600; border: none; border-radius: 8px; cursor: pointer; transition: opacity 0.15s; }
  button:hover { opacity: 0.9; }
  #startBtn { background: #16a34a; color: #fff; }
  #endBtn   { background: #dc2626; color: #fff; display: none; }
  .metrics { margin-top: 14px; font-size: 0.85em; color: #475569; }
  #log { margin-top: 20px; font-size: 0.82em; color: #334155; white-space: pre-wrap; max-height: 320px; overflow-y: auto; background: #f8fafc; padding: 14px; border-radius: 8px; border: 1px solid #e2e8f0; font-family: monospace; }
</style>
</head>
<body>
<div class="card">
  <h1>📞 Jain College MBA Admissions</h1>
  <div class="subtitle">AI Telephony Admission Counselor – Divya (Fast Latency Pipeline)</div>

  <div class="btn-group">
    <button id="startBtn" onclick="startCall()">📞 Start Call</button>
    <button id="endBtn"   onclick="endCall()">🔴 End Call</button>
  </div>

  <div id="status" class="idle">Press "Start Call" to begin speaking with Divya.</div>
  <div id="log"></div>
</div>

<script>
const SILENCE_MS       = 1050;   // Snappy ~1.05s silence window for natural conversational turn-taking
const RMS_THRESHOLD    = 0.012;  // Voice activity detection sensitivity
const SAMPLE_RATE      = 16000;
const CHUNK_MS         = 80;     // Analyser poll interval

let mediaStream        = null;
let audioContext       = null;
let analyser           = null;
let mediaRecorder      = null;
let recordedChunks     = [];
let silenceTimer       = null;
let speaking           = false;
let callActive         = false;
let divyaSpeaking      = false;
let sessionId          = null;

function log(msg) {
  const el = document.getElementById("log");
  el.textContent += "[" + new Date().toLocaleTimeString() + "] " + msg + "\\n";
  el.scrollTop = el.scrollHeight;
}

function setStatus(text, cls) {
  const el = document.getElementById("status");
  el.textContent = text;
  el.className = cls;
}

function generateSessionId() {
  return "call-" + Date.now() + "-" + Math.random().toString(36).slice(2, 8);
}

async function startCall() {
  sessionId = generateSessionId();
  log("Session initiated: " + sessionId);

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch (e) {
    log("Microphone access error: " + e);
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
  document.getElementById("startBtn").style.display = "none";
  document.getElementById("endBtn").style.display   = "inline-block";

  setStatus("🎙️ Listening... (Speak now)", "listening");
  log("Call connected. Listening for speech...");

  startListening();
}

function endCall() {
  callActive = false;
  stopListening();
  if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
  if (audioContext) audioContext.close();
  document.getElementById("startBtn").style.display = "inline-block";
  document.getElementById("endBtn").style.display   = "none";
  setStatus("Call ended.", "idle");
  log("Call ended.");
}

function startListening() {
  if (!callActive || divyaSpeaking) return;

  recordedChunks = [];
  speaking       = false;

  try {
    mediaRecorder = new MediaRecorder(mediaStream);
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
    mediaRecorder.onstop = onRecordingStop;
    mediaRecorder.start(CHUNK_MS);
  } catch (e) {
    log("MediaRecorder start error: " + e);
  }

  pollVAD();
}

function stopListening() {
  clearTimeout(silenceTimer);
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
}

function pollVAD() {
  if (!callActive || divyaSpeaking) return;

  const buf = new Float32Array(analyser.fftSize);
  analyser.getFloatTimeDomainData(buf);

  let sum = 0;
  for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
  const rms = Math.sqrt(sum / buf.length);

  if (rms > RMS_THRESHOLD) {
    if (!speaking) {
      speaking = true;
      clearTimeout(silenceTimer);
      log("Voice detected (RMS: " + rms.toFixed(4) + ")");
    } else {
      clearTimeout(silenceTimer);
    }
    silenceTimer = setTimeout(onSilence, SILENCE_MS);
  }

  setTimeout(pollVAD, CHUNK_MS);
}

function onSilence() {
  if (!speaking || !callActive || divyaSpeaking) return;
  log("Silence detected -> sending utterance to backend...");
  stopListening();
}

function onRecordingStop() {
  if (!speaking || recordedChunks.length === 0) {
    if (callActive && !divyaSpeaking) startListening();
    return;
  }

  const blob = new Blob(recordedChunks, { type: "audio/webm" });
  sendAudio(blob);
}

async function sendAudio(blob) {
  setStatus("⏳ Processing response...", "processing");
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
    const roundtrip = ((t1 - t0) / 1000).toFixed(3);
    log("[TIMING] ROUNDTRIP: " + roundtrip + "s");

    if (!response.ok) {
      log("Server error: " + response.status);
      setStatus("🎙️ Listening...", "listening");
      startListening();
      return;
    }

    const data = await response.json();

    if (data.transcript) log("Student: " + data.transcript);
    if (data.reply)      log("Divya:   " + data.reply);

    if (data.audio_b64) {
      await playAudio(data.audio_b64, data.sample_rate || 22050);
    } else {
      setStatus("🎙️ Listening...", "listening");
      startListening();
    }

  } catch (e) {
    log("Network fetch error: " + e);
    setStatus("🎙️ Listening...", "listening");
    startListening();
  }
}

async function playAudio(b64, sampleRate) {
  divyaSpeaking = true;
  setStatus("🔊 Divya is speaking...", "speaking");

  try {
    const binary = atob(b64);
    const bytes  = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

    const t_decode_start = performance.now();
    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }
    const decoded = await audioContext.decodeAudioData(bytes.buffer);
    const t_decode_elapsed = ((performance.now() - t_decode_start) / 1000).toFixed(3);
    log("[TIMING] AUDIO_DECODE: " + t_decode_elapsed + "s");

    const source = audioContext.createBufferSource();
    source.buffer = decoded;
    source.connect(audioContext.destination);

    source.onended = () => {
      divyaSpeaking = false;
      if (callActive) {
        setStatus("🎙️ Listening...", "listening");
        startListening();
      }
    };

    source.start();
  } catch (err) {
    log("Audio playback error: " + err);
    divyaSpeaking = false;
    if (callActive) {
      setStatus("🎙️ Listening...", "listening");
      startListening();
    }
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
