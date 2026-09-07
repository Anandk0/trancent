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
# HINGLISH → DEVANAGARI TTS NORMALIZER
# ============================================================

def normalize_for_tts(text: str) -> tuple:
    """Returns (normalized_text, elapsed_seconds)."""

    prompt = f"""
Convert the following conversational Hinglish/Hindi text into
Devanagari script so that a Hindi text-to-speech model can pronounce
it naturally.

IMPORTANT RULES:

1. Preserve the exact meaning.
2. Do NOT add information.
3. Do NOT remove information.
4. Do NOT answer or respond to the text.
5. Only convert the supplied text.
6. If the text is Hindi or Hinglish, write the ENTIRE response
   in Devanagari script.
7. Romanized Hindi words must be converted to natural Devanagari.
8. English words that are being spoken as part of Hindi/Hinglish
   must also be written phonetically in Devanagari.
9. Do NOT leave English words in Latin/Roman letters when converting
   Hindi/Hinglish.
10. Preserve abbreviations phonetically.

Examples:

MBA → एमबीए
HR → एचआर

Finance → फाइनेंस
Marketing → मार्केटिंग
Placement → प्लेसमेंट
Admission → एडमिशन
Specialization → स्पेशलाइज़ेशन
Business Analytics → बिज़नेस एनालिटिक्स
Help → हेल्प
Interest → इंटरेस्ट
Relax → रिलैक्स

11. Do NOT translate these professional terms into formal Hindi.
12. Preserve names in a pronounceable Devanagari form.
13. Preserve the original meaning and sentence structure.
14. Preserve punctuation where it represents a natural pause.
15. Do NOT add punctuation around individual English-derived words.
16. English-derived words written in Devanagari must flow naturally
    with the surrounding Hindi sentence.
17. Do not create pauses around individual English-derived words.
18. Keep connected phrases together so the TTS speaks naturally.
19. Return ONLY the converted Devanagari text.
20. Do not include explanations.
21. Do not include quotation marks.

Example:

Input:
Haan sir, MBA ka duration do saal ka hai.

Output:
हाँ सर, एमबीए का ड्यूरेशन दो साल का है।

Input:
Aapko Finance specialization mein interest hai?

Output:
आपको फाइनेंस स्पेशलाइज़ेशन में इंटरेस्ट है?

Input:
Placement ke baare mein jaana hai?

Output:
प्लेसमेंट के बारे में जानना है?

Input:
Business Analytics mein interest hai kya?

Output:
बिज़नेस एनालिटिक्स में इंटरेस्ट है क्या?

TEXT TO CONVERT:

{text}
"""

    t0 = time.perf_counter()

    try:

        response = requests.post(
            GEMMA_URL,
            json={
                # IMPORTANT:
                # Use a separate session so the TTS conversion
                # does not become part of the counselor's memory.
                "session_id": "tts-normalizer-" + str(uuid.uuid4()),
                "text": prompt
            },
            timeout=120
        )

        response.raise_for_status()

        result = response.json()

        normalized_text = result.get("reply", "").strip()

        elapsed = time.perf_counter() - t0

        if normalized_text:
            return normalized_text, elapsed

        return text, elapsed

    except Exception as e:

        elapsed = time.perf_counter() - t0
        print("TTS normalization error:", e)

        # Do not break the complete conversation if
        # normalization fails.
        return text, elapsed


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

    print("\n[1/4] Sending audio to ASR...")

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
    # 2. TRANSCRIPT → GEMMA COUNSELOR
    # ========================================================

    print("\n[2/4] Sending transcript to Gemma...")

    t_gemma_start = time.perf_counter()

    try:

        gemma_response = requests.post(
            GEMMA_URL,
            json={
                "session_id": session_id,
                "text": transcript
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

    reply = gemma_result.get("reply", "").strip()

    print("Gemma reply:")
    print(reply)


    # ========================================================
    # 3. HINGLISH → DEVANAGARI
    # ========================================================

    print("\n[3/4] Converting response for TTS...")

    tts_text, t_norm_elapsed = normalize_for_tts(reply)
    print(f"[TIMING] TTS_NORMALIZATION: {t_norm_elapsed:.3f}s")

    print("TTS text:")
    print(tts_text)


    # ========================================================
    # 4. DEVANAGARI → PARLER TTS
    # ========================================================

    print("\n[4/4] Sending text to Parler TTS...")

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
    print(f"[TIMING] AUDIO_READ:        {t_read_elapsed:.3f}s")
    print(f"[TIMING] ASR:               {t_asr_elapsed:.3f}s")
    print(f"[TIMING] GEMMA:             {t_gemma_elapsed:.3f}s")
    print(f"[TIMING] TTS_NORMALIZATION: {t_norm_elapsed:.3f}s")
    print(f"[TIMING] TTS:               {t_tts_elapsed:.3f}s")
    print(f"[TIMING] AGENT_TOTAL:       {t_total:.3f}s")
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