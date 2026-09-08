import io
import re
import json
import time
import base64
import uuid
from typing import AsyncGenerator

import requests
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse


# ============================================================
# SERVICE URLS & HTTP CONNECTION POOL
# ============================================================

ASR_URL = "http://127.0.0.1:8001/transcribe"
GEMMA_URL = "http://127.0.0.1:8000/chat"
TTS_URL = "http://127.0.0.1:8003/synthesize"
AGENT_URL = "http://127.0.0.1:8002/chat"

HOST = "0.0.0.0"
PORT = 8090

app = FastAPI(title="MBA Streaming Calling Agent")

# Persistent HTTP session for proxy-compatible internal forwarding
http_session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=15, pool_maxsize=30)
http_session.mount("http://", adapter)
http_session.mount("https://", adapter)

# In-memory progressive audio buffers for partial ASR streaming
session_audio_buffers: dict[str, bytearray] = {}
session_last_asr_time: dict[str, float] = {}


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "mba-streaming-calling-agent",
        "mode": "http-sse-streaming",
        "proxy_compatible": True
    }


# ============================================================
# TEXT SEGMENTER (Sentence Pipelining for TTS)
# ============================================================

def split_into_sentences(text: str) -> list[str]:
    """
    Split Hindi, Hinglish, and English text into natural sentence / clause chunks
    suitable for progressive TTS synthesis.
    Splits on Devanagari danda (।), period (.), question mark (?), exclamation mark (!),
    or newline, while keeping chunks coherent and natural.
    """
    if not text:
        return []
    raw_chunks = re.split(r'([।\.\?\!\n]+)', text)
    sentences = []
    current = ""
    for piece in raw_chunks:
        current += piece
        if re.search(r'[।\.\?\!\n]', piece):
            cleaned = current.strip()
            if cleaned and len(cleaned) > 1:
                sentences.append(cleaned)
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences if sentences else [text.strip()]


def build_structured_prompt(transcript: str) -> str:
    return f"""You are Divya, the AI admissions counselor for Regular MBA at Jain College of Engineering and Research, Udyambag, Belagavi (VTU affiliated, AICTE approved).

Respond directly to the student's message.
Keep your response concise, helpful, and natural for a phone call (1 to 3 sentences maximum).

Return your response as a single valid JSON object with exactly two keys:
- "reply": your natural conversational response matching the student's language style (English, Hindi, or Hinglish).
- "tts_text": the exact same response written in Devanagari script for the Hindi TTS voice.

Rules for tts_text:
- If the reply is in Hindi or Hinglish, write the ENTIRE tts_text in Devanagari script.
- English professional words in Hindi/Hinglish must be written phonetically in Devanagari (e.g., MBA → एमबीए, HR → एचआर, Finance → फाइनेंस, Marketing → मार्केटिंग, Placement → प्लेसमेंट, Admission → एडमिशन, Specialization → स्पेशलाइज़ेशन, Business Analytics → बिज़नेस एनालिटिक्स).
- If the reply is purely in English, tts_text can remain in English.

Specializations available: Marketing, Finance, Human Resource Management, Business Analytics.
Do NOT invent fees, placement percentages, salary figures, recruiter names, or unverified deadlines.

Return ONLY the JSON object. No extra text, no markdown backticks.

Example output:
{{"reply": "Haan, MBA ka duration 2 saal ka hai. Aap kaunsi specialization mein interested hain?", "tts_text": "हाँ, एमबीए का ड्यूरेशन 2 साल का है। आप कौनसी स्पेशलाइज़ेशन में इंटरेस्टेड हैं?"}}

Student message:
{transcript}"""


def parse_gemma_json(raw: str, fallback_reply: str = "") -> tuple:
    if not raw or not raw.strip():
        return fallback_reply, fallback_reply

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            reply = str(data.get("reply", "") or "").strip()
            tts_text = str(data.get("tts_text", "") or "").strip()
            if not reply:
                reply = fallback_reply
            if not tts_text:
                tts_text = reply
            return reply, tts_text
    except Exception:
        pass

    brace_match = re.search(r"(\{[\s\S]*\})", text)
    if brace_match:
        try:
            data = json.loads(brace_match.group(1))
            if isinstance(data, dict):
                reply = str(data.get("reply", "") or "").strip()
                tts_text = str(data.get("tts_text", "") or "").strip()
                if not reply:
                    reply = fallback_reply
                if not tts_text:
                    tts_text = reply
                return reply, tts_text
        except Exception:
            pass

    reply_match = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    tts_match = re.search(r'"tts_text"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    reply = reply_match.group(1).replace('\\"', '"').strip() if reply_match else ""
    tts_text = tts_match.group(1).replace('\\"', '"').strip() if tts_match else ""

    if reply or tts_text:
        return reply or fallback_reply, tts_text or reply or fallback_reply

    return text or fallback_reply, text or fallback_reply


# ============================================================
# PROGRESSIVE / PARTIAL ASR ENDPOINT
# ============================================================

@app.post("/stream/asr_chunk")
async def receive_asr_chunk(request: Request):
    """
    Accepts incremental audio chunks from the browser during active speech,
    accumulates in the session buffer, and returns partial transcript.
    """
    session_id = request.headers.get("X-Session-Id", "default-session")
    chunk_bytes = await request.body()

    if session_id not in session_audio_buffers:
        session_audio_buffers[session_id] = bytearray()
        session_last_asr_time[session_id] = 0.0

    session_audio_buffers[session_id].extend(chunk_bytes)
    now = time.perf_counter()

    # Throttle partial ASR inference to every 350ms to keep CPU load optimal
    if now - session_last_asr_time[session_id] < 0.35 and len(session_audio_buffers[session_id]) < 64000:
        return {"status": "buffering", "session_id": session_id, "bytes_accumulated": len(session_audio_buffers[session_id])}

    session_last_asr_time[session_id] = now
    buffer_copy = bytes(session_audio_buffers[session_id])

    try:
        t0 = time.perf_counter()
        asr_response = http_session.post(
            ASR_URL,
            files={"file": ("chunk.webm", buffer_copy, "audio/webm")},
            data={"language": "hi", "decoding": "ctc"},
            timeout=5
        )
        if asr_response.ok:
            data = asr_response.json()
            partial = data.get("text", "").strip()
            t_elapsed = time.perf_counter() - t0
            return {
                "status": "success",
                "session_id": session_id,
                "partial_transcript": partial,
                "asr_time_s": round(t_elapsed, 3),
                "bytes_accumulated": len(buffer_copy)
            }
    except Exception as e:
        pass

    return {"status": "buffering", "session_id": session_id, "bytes_accumulated": len(buffer_copy)}


# ============================================================
# STREAMING SSE PIPELINE (Turn Completion → Gemma → TTS Chunks)
# ============================================================

@app.post("/stream/chat_events")
async def chat_events_stream(
    file: UploadFile = File(None),
    transcript_override: str = Form(None),
    session_id: str = Form(None),
    language: str = Form("hi")
):
    if not session_id:
        session_id = str(uuid.uuid4())

    async def event_generator() -> AsyncGenerator[str, None]:
        t_turn_start = time.perf_counter()
        transcript = (transcript_override or "").strip()
        t_asr = 0.0

        # Step 1: ASR Transcription (if audio uploaded and no transcript override provided)
        if not transcript and file is not None:
            t_asr_start = time.perf_counter()
            audio_bytes = await file.read()
            # Reset session buffer
            if session_id in session_audio_buffers:
                session_audio_buffers[session_id] = bytearray()

            try:
                asr_response = http_session.post(
                    ASR_URL,
                    files={"file": (file.filename or "audio.wav", audio_bytes, file.content_type or "audio/wav")},
                    data={"language": language, "decoding": "ctc"},
                    timeout=120
                )
                asr_response.raise_for_status()
                asr_result = asr_response.json()
                transcript = asr_result.get("text", "").strip()
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'stage': 'asr', 'error': str(e)})}\n\n"
                return

            t_asr = time.perf_counter() - t_asr_start
            yield f"data: {json.dumps({'type': 'asr_final', 'transcript': transcript, 'asr_time_s': round(t_asr, 3)})}\n\n"

        if not transcript:
            yield f"data: {json.dumps({'type': 'done', 'reply': '', 'tts_text': '', 'audio_chunks': 0})}\n\n"
            return

        # Step 2: Gemma Generation
        yield f"data: {json.dumps({'type': 'gemma_start'})}\n\n"
        t_gemma_start = time.perf_counter()
        structured_prompt = build_structured_prompt(transcript)

        try:
            gemma_response = http_session.post(
                GEMMA_URL,
                json={"session_id": session_id, "text": structured_prompt},
                timeout=120
            )
            gemma_response.raise_for_status()
            gemma_result = gemma_response.json()
            raw_gemma = gemma_result.get("reply", "").strip()
            reply, tts_text = parse_gemma_json(raw_gemma, fallback_reply=transcript)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'stage': 'gemma', 'error': str(e)})}\n\n"
            return

        t_gemma_elapsed = time.perf_counter() - t_gemma_start
        yield f"data: {json.dumps({'type': 'gemma_final', 'reply': reply, 'tts_text': tts_text, 'gemma_time_s': round(t_gemma_elapsed, 3)})}\n\n"

        # Step 3: Sentence-Pipelined TTS Synthesis
        # Segment tts_text into sentence chunks and synthesize sequentially
        sentence_chunks = split_into_sentences(tts_text)
        first_audio_emitted = False
        t_first_audio_latency = 0.0

        for idx, sentence in enumerate(sentence_chunks):
            t_tts_start = time.perf_counter()
            try:
                tts_response = http_session.post(
                    TTS_URL,
                    json={"text": sentence},
                    timeout=120
                )
                tts_response.raise_for_status()
                tts_result = tts_response.json()
                audio_b64 = tts_result.get("audio_b64")
                sample_rate = tts_result.get("sample_rate", 22050)
                t_tts_elapsed = time.perf_counter() - t_tts_start

                if not first_audio_emitted:
                    first_audio_emitted = True
                    t_first_audio_latency = time.perf_counter() - t_turn_start

                chunk_payload = {
                    "type": "audio_chunk",
                    "chunk_index": idx,
                    "total_chunks": len(sentence_chunks),
                    "sentence": sentence,
                    "audio_b64": audio_b64,
                    "sample_rate": sample_rate,
                    "tts_time_s": round(t_tts_elapsed, 3),
                    "first_audio_latency_s": round(t_first_audio_latency, 3)
                }
                yield f"data: {json.dumps(chunk_payload)}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'tts_chunk_error', 'chunk_index': idx, 'error': str(e)})}\n\n"

        t_turn_total = time.perf_counter() - t_turn_start

        # Step 4: Done & Full Telemetry
        done_payload = {
            "type": "done",
            "session_id": session_id,
            "transcript": transcript,
            "reply": reply,
            "tts_text": tts_text,
            "sentence_chunks": len(sentence_chunks),
            "timings": {
                "asr_s": round(t_asr, 3),
                "gemma_s": round(t_gemma_elapsed, 3),
                "first_audio_latency_s": round(t_first_audio_latency, 3),
                "total_turn_s": round(t_turn_total, 3)
            }
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================================
# STREAMING BROWSER INTERFACE (HTML5 Web Audio Sequential Queue)
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=STREAMING_HTML_PAGE)


STREAMING_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Jain MBA Admissions – Real-Time Voice (Divya)</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 680px; margin: 30px auto; padding: 20px; background: #f8fafc; color: #0f172a; }
  .card { background: #ffffff; border-radius: 14px; padding: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); }
  h1 { font-size: 1.4em; margin-top: 0; color: #0f172a; }
  .subtitle { font-size: 0.9em; color: #64748b; margin-bottom: 18px; }
  #status { margin: 14px 0; padding: 14px 18px; border-radius: 8px; font-weight: 600; font-size: 0.95em; }
  #status.listening { background: #dcfce7; color: #15803d; border-left: 4px solid #22c55e; }
  #status.processing { background: #fef9c3; color: #a16207; border-left: 4px solid #eab308; }
  #status.speaking { background: #dbeafe; color: #1d4ed8; border-left: 4px solid #3b82f6; }
  #status.idle { background: #f1f5f9; color: #64748b; }
  .btn-group { display: flex; gap: 12px; margin-top: 12px; }
  button { padding: 12px 24px; font-size: 1em; font-weight: 600; border: none; border-radius: 8px; cursor: pointer; transition: opacity 0.15s; }
  #startBtn { background: #16a34a; color: #fff; }
  #endBtn   { background: #dc2626; color: #fff; display: none; }
  .box { margin-top: 16px; padding: 12px 16px; border-radius: 8px; background: #f8fafc; border: 1px solid #e2e8f0; }
  .box-title { font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; color: #64748b; margin-bottom: 6px; }
  #partialASR { font-size: 1.05em; color: #0284c7; min-height: 24px; font-style: italic; }
  #counselorReply { font-size: 1.05em; color: #0f172a; min-height: 24px; }
  .telemetry { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 16px; }
  .metric-card { background: #f1f5f9; padding: 10px 14px; border-radius: 8px; text-align: center; }
  .metric-val { font-size: 1.2em; font-weight: 700; color: #0f172a; }
  .metric-lbl { font-size: 0.72em; color: #64748b; text-transform: uppercase; margin-top: 2px; }
  #log { margin-top: 20px; font-size: 0.8em; color: #334155; white-space: pre-wrap; max-height: 240px; overflow-y: auto; background: #0f172a; color: #f8fafc; padding: 14px; border-radius: 8px; font-family: monospace; }
</style>
</head>
<body>
<div class="card">
  <h1>📞 Jain College MBA Admissions</h1>
  <div class="subtitle">AI Telephony Admission Counselor – Divya (Real-Time Streaming Voice)</div>

  <div class="btn-group">
    <button id="startBtn" onclick="startCall()">📞 Start Call</button>
    <button id="endBtn"   onclick="endCall()">🔴 End Call</button>
  </div>

  <div id="status" class="idle">Press "Start Call" to begin speaking with Divya.</div>

  <div class="box">
    <div class="box-title">Live Partial Speech Recognition (ASR)</div>
    <div id="partialASR">Listening...</div>
  </div>

  <div class="box">
    <div class="box-title">Divya (Admission Counselor)</div>
    <div id="counselorReply">—</div>
  </div>

  <div class="telemetry">
    <div class="metric-card">
      <div id="mFirstAudio" class="metric-val">—</div>
      <div class="metric-lbl">First Audio Delay</div>
    </div>
    <div class="metric-card">
      <div id="mChunks" class="metric-val">—</div>
      <div class="metric-lbl">Sentence Chunks</div>
    </div>
    <div class="metric-card">
      <div id="mTotalTurn" class="metric-val">—</div>
      <div class="metric-lbl">Turn Time</div>
    </div>
  </div>

  <div id="log"></div>
</div>

<script>
const SILENCE_MS       = 1200;   // Turn completion silence threshold
const RMS_THRESHOLD    = 0.012;  // Voice activity threshold
const SAMPLE_RATE      = 16000;
const CHUNK_MS         = 100;    // Analyser poll interval
const ASR_STREAM_MS    = 350;    // Interval to send partial audio chunks for live ASR

let mediaStream        = null;
let audioContext       = null;
let analyser           = null;
let mediaRecorder      = null;
let recordedChunks     = [];
let silenceTimer       = null;
let vadTimeoutId       = null;
let asrIntervalId      = null;
let speaking           = false;
let callActive         = false;
let divyaSpeaking      = false;
let isProcessing       = false;
let sessionId          = null;

// Sequential Audio Chunk Playback Queue
let audioQueue         = [];
let isPlayingQueue     = false;
let currentSource      = null;

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
  return "stream-" + Date.now() + "-" + Math.random().toString(36).slice(2, 8);
}

async function startCall() {
  sessionId = generateSessionId();
  log("Streaming session: " + sessionId);

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

  document.getElementById("startBtn").style.display = "none";
  document.getElementById("endBtn").style.display   = "inline-block";

  setStatus("🎙️ Listening... (Speak naturally)", "listening");
  log("Call connected. Real-time streaming active.");

  startListening();
}

function endCall() {
  callActive = false;
  isProcessing = false;
  divyaSpeaking = false;
  stopListening();
  if (currentSource) {
    try { currentSource.stop(); } catch(e) {}
  }
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
  clearTimeout(silenceTimer);
  silenceTimer = null;
  clearTimeout(vadTimeoutId);
  vadTimeoutId = null;

  document.getElementById("partialASR").textContent = "Listening...";

  try {
    mediaRecorder = new MediaRecorder(mediaStream);
    mediaRecorder.ondataavailable = onDataAvailable;
    mediaRecorder.onstop = onRecordingStop;
    mediaRecorder.start(CHUNK_MS);
  } catch (e) {
    log("MediaRecorder start error: " + e);
    return;
  }

  pollVAD();
}

function onDataAvailable(e) {
  if (e.data.size > 0) {
    recordedChunks.push(e.data);
    // Send partial chunk to ASR if currently speaking
    if (speaking && !isProcessing) {
      sendPartialChunk(e.data);
    }
  }
}

async function sendPartialChunk(blobChunk) {
  try {
    const arrayBuf = await blobChunk.arrayBuffer();
    const res = await fetch("stream/asr_chunk", {
      method: "POST",
      headers: { "X-Session-Id": sessionId, "Content-Type": "audio/webm" },
      body: arrayBuf
    });
    if (res.ok) {
      const data = await res.json();
      if (data.partial_transcript) {
        document.getElementById("partialASR").textContent = data.partial_transcript + " ...";
      }
    }
  } catch (e) {}
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
      log("Voice activity detected (RMS " + rms.toFixed(4) + ")");
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

  isProcessing = true;
  log("Silence detected – completed user turn. Streaming response...");
  stopListening();
}

function onRecordingStop() {
  if (!speaking || recordedChunks.length === 0) {
    isProcessing = false;
    if (callActive && !divyaSpeaking) startListening();
    return;
  }

  const blob = new Blob(recordedChunks, { type: "audio/webm" });
  submitTurnStream(blob);
}

async function submitTurnStream(blob) {
  setStatus("⏳ Processing & streaming Divya audio...", "processing");
  const t_submit = performance.now();

  try {
    const formData = new FormData();
    formData.append("file", blob, "turn.webm");
    formData.append("session_id", sessionId);
    formData.append("language", "hi");

    const response = await fetch("stream/chat_events", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      log("Stream connection error: " + response.status);
      resetToListening();
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
              handleStreamEvent(event, t_submit);
            } catch (err) {
              console.error("JSON parse error on SSE:", err, jsonStr);
            }
          }
        }
      }
    }

  } catch (err) {
    log("Stream fetch error: " + err);
    resetToListening();
  }
}

function handleStreamEvent(event, t_submit) {
  if (event.type === "asr_final") {
    document.getElementById("partialASR").textContent = event.transcript || "(No speech detected)";
    log("Student: " + event.transcript + " [ASR: " + event.asr_time_s + "s]");
  }
  else if (event.type === "gemma_final") {
    document.getElementById("counselorReply").textContent = event.reply;
    log("Divya: " + event.reply + " [Gemma: " + event.gemma_time_s + "s]");
  }
  else if (event.type === "audio_chunk") {
    log("Received Audio Chunk " + (event.chunk_index + 1) + "/" + event.total_chunks +
        " (First Audio: " + event.first_audio_latency_s + "s, TTS: " + event.tts_time_s + "s)");
    document.getElementById("mFirstAudio").textContent = event.first_audio_latency_s + "s";
    document.getElementById("mChunks").textContent = (event.chunk_index + 1) + " / " + event.total_chunks;

    // Enqueue audio chunk for immediate sequential playback
    enqueueAudioChunk(event.audio_b64, event.sample_rate || 22050);
  }
  else if (event.type === "done") {
    if (event.timings) {
      document.getElementById("mTotalTurn").textContent = event.timings.total_turn_s + "s";
      log("[DONE] Turn Complete: First Audio " + event.timings.first_audio_latency_s + "s | Turn Total " + event.timings.total_turn_s + "s");
    }
  }
}

async function enqueueAudioChunk(b64, sampleRate) {
  if (!b64) return;

  try {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }
    const decoded = await audioContext.decodeAudioData(bytes.buffer);
    audioQueue.push(decoded);

    if (!isPlayingQueue) {
      playNextQueueChunk();
    }
  } catch (err) {
    console.error("Audio chunk decode error:", err);
  }
}

function playNextQueueChunk() {
  if (audioQueue.length === 0) {
    isPlayingQueue = false;
    divyaSpeaking = false;
    isProcessing = false;
    if (callActive) {
      setStatus("🎙️ Listening... (Speak now)", "listening");
      log("Audio playback finished. Resuming listening...");
      startListening();
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

  source.onended = () => {
    playNextQueueChunk();
  };

  source.start();
}

function resetToListening() {
  isProcessing = false;
  divyaSpeaking = false;
  if (callActive) {
    setStatus("🎙️ Listening...", "listening");
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
