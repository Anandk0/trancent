import re
import json
import time
import requests
import uuid

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn


# ============================================================
# SERVICE URLS & HTTP SESSION (Connection Pooling)
# ============================================================

ASR_URL = "http://127.0.0.1:8001/transcribe"
GEMMA_URL = "http://127.0.0.1:8000/chat"
TTS_URL = "http://127.0.0.1:8003/synthesize"

HOST = "0.0.0.0"
PORT = 8002

# Persistent HTTP session to reuse connections to local services
http_session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20)
http_session.mount("http://", adapter)
http_session.mount("https://", adapter)


app = FastAPI(title="MBA Calling Agent API")


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "mba-calling-agent"
    }


# ============================================================
# STRUCTURED GEMMA PROMPT
# ============================================================

# Wraps the user transcript so that one Gemma call returns both
# the conversational reply and the Devanagari TTS text.
# Keep responses concise (1 to 3 sentences) to minimize generation latency.

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
- Do NOT include unnecessary punctuation around English words.

Specializations available: Marketing, Finance, Human Resource Management, Business Analytics.
Do NOT invent fees, placement percentages, salary figures, recruiter names, or unverified deadlines.

Return ONLY the JSON object. No extra text, no markdown backticks.

Example output:
{{"reply": "Haan, MBA ka duration 2 saal ka hai. Aap kaunsi specialization mein interested hain?", "tts_text": "हाँ, एमबीए का ड्यूरेशन 2 साल का है। आप कौनसी स्पेशलाइज़ेशन में इंटरेस्टेड हैं?"}}

Student message:
{transcript}"""


# ============================================================
# PARSE GEMMA STRUCTURED RESPONSE
# ============================================================

def parse_gemma_json(raw: str, fallback_reply: str = "") -> tuple:
    """
    Parse the JSON object from Gemma's raw output.
    Returns (reply, tts_text).
    """
    if not raw or not raw.strip():
        return fallback_reply, fallback_reply

    text = raw.strip()

    # Strip markdown code fences if present e.g. ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # 1. Try standard JSON parse
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
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 2. Try extracting JSON object substring {...}
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

    # 3. Try regex extraction for "reply" and "tts_text"
    reply_match = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    tts_match = re.search(r'"tts_text"\s*:\s*"((?:[^"\\]|\\.)*)"', text)

    reply = ""
    tts_text = ""

    if reply_match:
        reply = reply_match.group(1).replace('\\"', '"').replace('\\n', ' ').strip()
    if tts_match:
        tts_text = tts_match.group(1).replace('\\"', '"').replace('\\n', ' ').strip()

    if reply or tts_text:
        if not reply:
            reply = fallback_reply
        if not tts_text:
            tts_text = reply
        print("[WARN] JSON parse recovered fields via regex.")
        return reply, tts_text

    # Complete fallback
    reply = text or fallback_reply
    print("[WARN] JSON parse failed; using raw Gemma output.")
    return reply, reply


# ============================================================
# MAIN CHAT ENDPOINT
# ============================================================

@app.post("/chat")
async def chat(
    file: UploadFile = File(...),
    session_id: str = Form(None),
    language: str = Form("hi")
):
    if not session_id:
        session_id = str(uuid.uuid4())

    t_request_start = time.perf_counter()

    print("\n" + "=" * 60)
    print("NEW CHAT REQUEST")
    print("Session:", session_id)
    print("=" * 60)

    # --------------------------------------------------------
    # 1. AUDIO → ASR
    # --------------------------------------------------------
    print("\n[1/3] Sending audio to ASR...")
    t_read_start = time.perf_counter()
    audio_data = await file.read()
    t_read_elapsed = time.perf_counter() - t_read_start

    t_asr_start = time.perf_counter()
    try:
        asr_response = http_session.post(
            ASR_URL,
            files={
                "file": (
                    file.filename or "audio.wav",
                    audio_data,
                    file.content_type or "audio/wav"
                )
            },
            data={
                "language": language,
                "decoding": "ctc"
            },
            timeout=120
        )
        asr_response.raise_for_status()
        asr_result = asr_response.json()
    except Exception as e:
        print("ASR ERROR:", e)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "stage": "asr",
                "error": str(e)
            }
        )

    t_asr_elapsed = time.perf_counter() - t_asr_start
    transcript = asr_result.get("text", "").strip()
    print(f"[TIMING] ASR: {t_asr_elapsed:.3f}s -> Transcript: '{transcript}'")

    # Empty transcript handling
    if not transcript:
        return {
            "status": "success",
            "session_id": session_id,
            "language": language,
            "transcript": "",
            "reply": "",
            "tts_text": "",
            "audio_file": None,
            "audio_b64": None
        }

    # --------------------------------------------------------
    # 2. TRANSCRIPT → GEMMA (Single-Pass Structured Output)
    # --------------------------------------------------------
    print("\n[2/3] Sending transcript to Gemma...")
    structured_prompt = build_structured_prompt(transcript)

    t_gemma_start = time.perf_counter()
    try:
        gemma_response = http_session.post(
            GEMMA_URL,
            json={
                "session_id": session_id,
                "text": structured_prompt
            },
            timeout=120
        )
        gemma_response.raise_for_status()
        gemma_result = gemma_response.json()
    except Exception as e:
        print("GEMMA ERROR:", e)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "stage": "gemma",
                "transcript": transcript,
                "error": str(e)
            }
        )

    t_gemma_elapsed = time.perf_counter() - t_gemma_start
    raw_gemma = gemma_result.get("reply", "").strip()
    reply, tts_text = parse_gemma_json(raw_gemma, fallback_reply=transcript)

    print(f"[TIMING] GEMMA: {t_gemma_elapsed:.3f}s")
    print(f"  Reply:    '{reply}'")
    print(f"  TTS text: '{tts_text}'")

    # --------------------------------------------------------
    # 3. DEVANAGARI → PARLER TTS
    # --------------------------------------------------------
    print("\n[3/3] Sending text to Parler TTS...")
    t_tts_start = time.perf_counter()
    try:
        tts_response = http_session.post(
            TTS_URL,
            json={
                "text": tts_text
            },
            timeout=120
        )
        tts_response.raise_for_status()
        tts_result = tts_response.json()
    except Exception as e:
        print("TTS ERROR:", e)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "stage": "tts",
                "session_id": session_id,
                "transcript": transcript,
                "reply": reply,
                "tts_text": tts_text,
                "error": str(e)
            }
        )

    t_tts_elapsed = time.perf_counter() - t_tts_start
    audio_file = tts_result.get("audio_file")
    audio_b64 = tts_result.get("audio_b64")
    sample_rate = tts_result.get("sample_rate", 22050)

    t_total = time.perf_counter() - t_request_start

    print("\n" + "=" * 60)
    print("REQUEST COMPLETE")
    print(f"Session:     {session_id}")
    print(f"Audio bytes: {len(audio_data)}")
    print(f"ASR:         {t_asr_elapsed:.3f} s")
    print(f"Gemma:       {t_gemma_elapsed:.3f} s")
    print(f"TTS:         {t_tts_elapsed:.3f} s")
    print(f"I/O:         {t_read_elapsed:.3f} s")
    print(f"TOTAL:       {t_total:.3f} s")
    print("=" * 60 + "\n")

    return {
        "status": "success",
        "session_id": session_id,
        "language": language,
        "transcript": transcript,
        "reply": reply,
        "tts_text": tts_text,
        "audio_file": audio_file,
        "audio_b64": audio_b64,
        "sample_rate": sample_rate,
        "timings": {
            "asr_s": round(t_asr_elapsed, 3),
            "gemma_s": round(t_gemma_elapsed, 3),
            "tts_s": round(t_tts_elapsed, 3),
            "total_s": round(t_total, 3)
        }
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT
    )
