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
# Browser POSTs raw WAV bytes here.
# Returns base64-encoded WAV audio from Divya.
# ============================================================

@app.post("/process")
async def process(request: Request):

    t_request_start = time.perf_counter()

    print("\n")
    print("=" * 60)
    print("CALL SERVER: NEW REQUEST")

    # --------------------------------------------------------
    # Read raw audio bytes from browser
    # --------------------------------------------------------

    t_read_start = time.perf_counter()
    audio_bytes = await request.body()
    t_read_elapsed = time.perf_counter() - t_read_start

    print(f"[TIMING] BROWSER_AUDIO_READ: {t_read_elapsed:.3f}s  ({len(audio_bytes)} bytes)")

    if not audio_bytes:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": "No audio received"}
        )

    session_id = request.headers.get("X-Session-Id", str(uuid.uuid4()))
    print("Session:", session_id)

    # --------------------------------------------------------
    # Forward audio to agent_api /chat
    # --------------------------------------------------------

    t_agent_start = time.perf_counter()

    try:
        agent_response = requests.post(
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

    # --------------------------------------------------------
    # Read generated WAV file and encode as base64
    # --------------------------------------------------------

    audio_file = agent_result.get("audio_file")

    t_encode_start = time.perf_counter()

    audio_b64 = None
    if audio_file:
        try:
            with open(audio_file, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print("Audio file read error:", e)

    t_encode_elapsed = time.perf_counter() - t_encode_start
    print(f"[TIMING] FILE_READ_ENCODE: {t_encode_elapsed:.3f}s")

    t_total = time.perf_counter() - t_request_start

    print(f"[TIMING] CALL_SERVER_TOTAL: {t_total:.3f}s")
    print("=" * 60)

    return JSONResponse(content={
        "status": "success",
        "session_id": session_id,
        "transcript": agent_result.get("transcript", ""),
        "reply": agent_result.get("reply", ""),
        "audio_b64": audio_b64,
        "sample_rate": 22050
    })


# ============================================================
# BROWSER PAGE
# Single-page call interface with:
# - browser microphone via MediaRecorder
# - RMS-based voice activity detection
# - silence detection (~1.2 s)
# - HTTP POST to /process
# - base64 WAV playback
# - microphone muted while Divya is speaking
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
  #log { margin-top: 20px; font-size: 0.82em; color: #555; white-space: pre-wrap; max-height: 300px; overflow-y: auto; background: #fff; padding: 10px; border-radius: 6px; border: 1px solid #ddd; }
</style>
</head>
<body>
<h1>📞 Jain College MBA Admissions</h1>
<p>AI Admission Counselor – Divya</p>

<button id="startBtn" onclick="startCall()">📞 Start Call</button>
<button id="endBtn"   onclick="endCall()">🔴 End Call</button>

<div id="status" class="idle">Press Start Call to begin.</div>
<div id="log"></div>

<script>
const SILENCE_MS       = 1200;   // ms of silence before sending
const RMS_THRESHOLD    = 0.012;  // voice activity threshold
const SAMPLE_RATE      = 16000;
const CHUNK_MS         = 100;    // analyser poll interval

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
  el.textContent += new Date().toLocaleTimeString() + "  " + msg + "\\n";
  el.scrollTop = el.scrollHeight;
}

function setStatus(text, cls) {
  const el = document.getElementById("status");
  el.textContent = text;
  el.className = cls;
}

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

  audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
  const source = audioContext.createMediaStreamSource(mediaStream);

  analyser = audioContext.createAnalyser();
  analyser.fftSize = 512;
  source.connect(analyser);

  callActive = true;
  document.getElementById("startBtn").style.display = "none";
  document.getElementById("endBtn").style.display   = "inline-block";

  setStatus("🎙️ Listening...", "listening");
  log("Call started.");

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

  mediaRecorder = new MediaRecorder(mediaStream);
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
  mediaRecorder.onstop = onRecordingStop;
  mediaRecorder.start(100);

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
      log("Voice detected (RMS " + rms.toFixed(4) + ")");
    } else {
      clearTimeout(silenceTimer);
    }
    silenceTimer = setTimeout(onSilence, SILENCE_MS);
  }

  setTimeout(pollVAD, CHUNK_MS);
}

function onSilence() {
  if (!speaking || !callActive || divyaSpeaking) return;
  log("Silence detected – sending audio...");
  stopListening();
}

function onRecordingStop() {
  if (!speaking || recordedChunks.length === 0) {
    if (callActive && !divyaSpeaking) startListening();
    return;
  }

  const blob = new Blob(recordedChunks, { type: "audio/webm" });
  log("Audio blob: " + blob.size + " bytes");
  sendAudio(blob);
}

async function sendAudio(blob) {
  setStatus("⏳ Processing...", "processing");

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
      setStatus("🎙️ Listening...", "listening");
      startListening();
      return;
    }

    const data = await response.json();

    if (data.transcript) log("You: " + data.transcript);
    if (data.reply)      log("Divya: " + data.reply);

    if (data.audio_b64) {
      const t2 = performance.now();
      await playAudio(data.audio_b64, data.sample_rate || 22050);
      const t3 = performance.now();
      log("[TIMING] BROWSER_AUDIO_PLAY: " + ((t3 - t2) / 1000).toFixed(3) + "s");
    } else {
      setStatus("🎙️ Listening...", "listening");
      startListening();
    }

  } catch (e) {
    log("Fetch error: " + e);
    setStatus("🎙️ Listening...", "listening");
    startListening();
  }
}

async function playAudio(b64, sampleRate) {
  divyaSpeaking = true;
  setStatus("🔊 Divya is speaking...", "speaking");

  const binary  = atob(b64);
  const bytes   = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

  const t_decode_start = performance.now();
  const decoded = await audioContext.decodeAudioData(bytes.buffer);
  const t_decode_elapsed = (performance.now() - t_decode_start) / 1000;
  log("[TIMING] BROWSER_AUDIO_DECODE: " + t_decode_elapsed.toFixed(3) + "s");

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
