"""
Svara TTS Adapter — Parler-compatible drop-in.

Exposes the SAME /synthesize contract that agent_api.py already calls
(POST {"text": ...} -> {"audio_b64", "sample_rate"}), but internally forwards
to the Svara TTS server's OpenAI-compatible /v1/audio/speech endpoint.

This lets us replace the slow autoregressive Parler TTS with fast Svara WITHOUT
changing agent_api.py: agent_api keeps POSTing sentences to 127.0.0.1:8003 and
gets back base64 WAV, exactly as before.

Run this on port 8003 (where Parler used to run). It is a lightweight HTTP
proxy — no GPU, no torch — so it can run in any env that has fastapi + requests.
The actual Svara model server must be running separately (default: port 8095).
"""

import base64
import os

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel


# ============================================================
# CONFIG
# ============================================================

# Svara model server (OpenAI-compatible speech endpoint).
#
# Configurable because Svara no longer has to be local. Running it in its
# own pod on its own MIG slice is the point: sharing one slice with Gemma,
# each was starving the other (Gemma 34 tok/s alone vs ~3 tok/s during a
# call; Svara 0.96s per clause alone vs 2.6-7.8s during a call). Pointing
# this at another host costs well under a millisecond inside the same
# cluster -- nothing next to the seconds the contention was costing.
#
#   SVARA_HOST=svaara-0 python svara_tts_api.py
SVARA_HOST = os.environ.get("SVARA_HOST", "127.0.0.1")
SVARA_PORT = os.environ.get("SVARA_PORT", "8095")
SVARA_URL = os.environ.get(
    "SVARA_URL", f"http://{SVARA_HOST}:{SVARA_PORT}/v1/audio/speech"
)

HOST = os.environ.get("SVARA_ADAPTER_HOST", "0.0.0.0")
# 8003 is the port agent_api calls. Default back to it: the Kokoro swap is
# not in place, so this adapter owns that port again.
PORT = int(os.environ.get("SVARA_ADAPTER_PORT", "8003"))

# The counselor's spoken reply is Devanagari (Hindi/Hinglish), so hi_female is
# the default. Language routing is available if agent_api ever passes a
# language hint.
DEFAULT_VOICE = "hi_female"
LANG_TO_VOICE = {
    "hi": "hi_female",
    "kn": "kn_female",
    "mr": "mr_female",
    "en": "hi_female",  # Indian-English handled well by the Hindi voice
}

# Svara outputs 24 kHz mono WAV.
SAMPLE_RATE = 24000

# Reuse one connection to the Svara server.
http_session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20)
http_session.mount("http://", adapter)

app = FastAPI(title="Svara TTS Adapter (Parler-compatible)")


class TTSRequest(BaseModel):
    text: str
    language: str = None
    voice: str = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "svara-tts-adapter",
        "backend": SVARA_URL,
        "default_voice": DEFAULT_VOICE,
    }


@app.post("/synthesize")
def synthesize(request: TTSRequest):
    text = (request.text or "").strip()
    if not text:
        return {"status": "error", "message": "Text cannot be empty"}

    # Pick the voice: explicit voice > language mapping > default.
    voice = request.voice or LANG_TO_VOICE.get(
        (request.language or "").lower(), DEFAULT_VOICE
    )

    try:
        resp = http_session.post(
            SVARA_URL,
            json={
                "input": text,
                "voice": voice,
                "response_format": "wav",
            },
            timeout=120,
        )
        resp.raise_for_status()
        audio_bytes = resp.content
    except Exception as e:
        print("SVARA TTS ERROR:", e)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "stage": "svara", "error": str(e)},
        )

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    return {
        "status": "success",
        "text": text,
        "voice": voice,
        "audio_b64": audio_b64,
        "sample_rate": SAMPLE_RATE,
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
