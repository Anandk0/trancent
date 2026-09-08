import re
import json
import time
import requests
import uuid

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn


# ============================================================
# SERVICE URLS
# ============================================================

ASR_URL = "http://127.0.0.1:8001/transcribe"
GEMMA_URL = "http://127.0.0.1:8000/chat"
TTS_URL = "http://127.0.0.1:8003/synthesize"

HOST = "0.0.0.0"
PORT = 8002


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
#
# The outer instruction is kept minimal so it does not inflate
# the prompt or confuse the counselor persona.

def build_structured_prompt(transcript: str) -> str:
    return f"""Respond as the MBA admissions counselor.

Reply to the student's message below.

Return your response as a single JSON object with exactly two keys:
- "reply": your natural conversational response (same language style as the student)
- "tts_text": the same response written entirely in Devanagari script for a Hindi TTS system

Rules for tts_text:
- If the reply is Hindi or Hinglish, write the ENTIRE tts_text in Devanagari.
- Romanized Hindi words must be converted to natural Devanagari.
- English professional terms spoken in a Hindi/Hinglish context must be written phonetically in Devanagari.
- Do NOT leave English words in Latin letters inside tts_text when the reply is Hindi/Hinglish.
- If the reply is fully in English, tts_text may remain in English.
- Do NOT add punctuation around individual English-derived Devanagari words.
- Preserve natural sentence flow so the TTS sounds like a real phone call.

Examples of phonetic Devanagari for common terms:
MBA → एमबीए, HR → एचआर, Finance → फाइनेंस, Marketing → मार्केटिंग,
Placement → प्लेसमेंट, Admission → एडमिशन, Specialization → स्पेशलाइज़ेशन,
Business Analytics → बिज़नेस एनालिटिक्स

Return ONLY the JSON object. No markdown. No code fences. No explanation.

Example output:
{{"reply": "Haan, MBA ka duration do saal ka hai.", "tts_text": "हाँ, एमबीए का ड्यूरेशन दो साल का है।"}}

Student message:
{transcript}"""


# ============================================================
# PARSE GEMMA STRUCTURED RESPONSE
# ============================================================

def parse_gemma_json(raw: str, fallback_reply: str = "") -> tuple:
    """
    Parse the JSON object from Gemma's raw output.

    Returns (reply, tts_text).

    Handles:
    - Clean JSON
    - JSON wrapped in markdown code fences (with or without surrounding text)
    - Partial/malformed JSON where only reply is present

    Fallback (no second Gemma call):
    - If tts_text is missing, use reply as tts_text.
    - If reply is missing, use fallback_reply.
    - Never raises; always returns two strings.
    """
    if not raw or not raw.strip():
        return fallback_reply, fallback_reply

    text = raw.strip()
    candidates = [text]

    # If wrapped in markdown code fences anywhere in text
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence_match:
        candidates.insert(0, fence_match.group(1).strip())

    # If there's an explicit JSON object {...} in text
    brace_match = re.search(r"(\{[\s\S]*\})", text)
    if brace_match:
        candidates.insert(0, brace_match.group(1).strip())

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                reply = str(data.get("reply", "") or "").strip()
                tts_text = str(data.get("tts_text", "") or "").strip()

                if not reply and not tts_text:
                    continue

                if not reply:
                    reply = fallback_reply
                if not tts_text:
                    tts_text = reply

                return reply, tts_text
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    # JSON parse failed — try regex extraction for "reply" and "tts_text"
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
        print("[WARN] JSON parse failed; extracted fields via regex.")
        return reply, tts_text

    # Complete parse failure — return the raw text as both fields
    # so the call does not crash.
    reply = text or fallback_reply
    print("[WARN] JSON parse completely failed; using raw Gemma output as reply and tts_text.")
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

    # --------------------------------------------------------
    # CREATE SESSION
    # --------------------------------------------------------

    if not session_id:
        session_id = str(uuid.uuid4())

    t_request_start = time.perf_counter()

    print("\n")
    print("=" * 60)
    print("NEW CHAT REQUEST")
    print("Session:", session_id)
    print("=" * 60)


    # ========================================================
    # 1. AUDIO → ASR
    # ========================================================

    print("\n[1/3] Sending audio to ASR...")

    t_read_start = time.perf_counter()
    audio_data = await file.read()
    t_read_elapsed = time.perf_counter() - t_read_start
    print(f"[TIMING] AUDIO_READ: {t_read_elapsed:.3f}s  ({len(audio_data)} bytes)")

    t_asr_start = time.perf_counter()

    try:

        asr_response = requests.post(
            ASR_URL,
            files={
                "file": (
                    file.filename,
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
    print(f"[TIMING] ASR: {t_asr_elapsed:.3f}s")

    transcript = asr_result.get("text", "").strip()

    print("ASR transcript:", transcript)


    # Empty transcript
    if not transcript:

        return {
            "status": "success",
            "session_id": session_id,
            "language": language,
            "transcript": "",
            "reply": "",
            "tts_text": "",
            "audio_file": None
        }


    # ========================================================
    # 2. TRANSCRIPT → GEMMA (single call, structured output)
    # ========================================================

    print("\n[2/3] Sending transcript to Gemma...")

    structured_prompt = build_structured_prompt(transcript)

    t_gemma_start = time.perf_counter()

    try:

        gemma_response = requests.post(
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
    print(f"[TIMING] GEMMA: {t_gemma_elapsed:.3f}s")

    raw_gemma = gemma_result.get("reply", "").strip()

    print("Gemma raw output:")
    print(raw_gemma)

    reply, tts_text = parse_gemma_json(raw_gemma, fallback_reply=transcript)

    print("reply:", reply)
    print("tts_text:", tts_text)


    # ========================================================
    # 3. DEVANAGARI → PARLER TTS
    # ========================================================

    print("\n[3/3] Sending text to Parler TTS...")

    t_tts_start = time.perf_counter()

    try:

        tts_response = requests.post(
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
    print(f"[TIMING] TTS: {t_tts_elapsed:.3f}s")

    audio_file = tts_result.get("audio_file")

    print("Audio file:", audio_file)

    t_total = time.perf_counter() - t_request_start

    print("\n")
    print("=" * 60)
    print("REQUEST COMPLETE")
    print(f"[TIMING] AUDIO_READ:  {t_read_elapsed:.3f}s")
    print(f"[TIMING] ASR:         {t_asr_elapsed:.3f}s")
    print(f"[TIMING] GEMMA:       {t_gemma_elapsed:.3f}s")
    print(f"[TIMING] TTS:         {t_tts_elapsed:.3f}s")
    print(f"[TIMING] AGENT_TOTAL: {t_total:.3f}s")
    print("=" * 60)


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {
        "status": "success",
        "session_id": session_id,
        "language": language,
        "transcript": transcript,
        "reply": reply,
        "tts_text": tts_text,
        "audio_file": audio_file
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
