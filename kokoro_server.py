"""
Kokoro TTS server — drop-in replacement for the Svara adapter on port 8003.

WHY THIS EXISTS
---------------
Measured on live calls, FIRST_AUDIO_READY == GEMMA_FIRST_SENTENCE +
TTS_FIRST_CHUNK exactly, and the sum stayed ~10-13s while the split between
the two swung wildly (Gemma 2.3s/TTS 7.8s on one turn, Gemma 10.9s/TTS 2.6s
on the next). That is Gemma (4B) and Svara (3B) fighting over the same MIG
slice.

Svara is Orpheus-style AUTOREGRESSIVE: it emits ~85 audio tokens per second
of speech, one at a time, on that contended GPU. Kokoro-82M is
StyleTTS2-based and NON-autoregressive — it produces the whole waveform in a
single forward pass at RTF ~0.03, so a 3-second clause takes ~100ms. It does
not just do the same work faster; it removes that entire class of work, and
an 82M model barely touches the GPU, which hands Gemma back its own
throughput.

CONTRACT
--------
Identical to svara_tts_api.py, so agent_api.py needs NO change:
    POST /synthesize  {"text": ...}  ->  {"audio_b64": ..., "sample_rate": ...}

ROUTING
-------
Kokoro covers Hindi + English, not Kannada/Marathi. Text is routed by script:
  - Kannada script            -> Svara fallback
  - Devanagari / Latin / else -> Kokoro
Any Kokoro failure also falls back to Svara, so this can never be a hard
regression versus today.

RUN
---
    python kokoro_server.py          # listens on 8003 (what agent_api calls)
Keep the Svara adapter running on SVARA_FALLBACK_PORT (8007) for fallback.
See KOKORO_SETUP.md.
"""

import base64
import io
import os
import threading
import time

import numpy as np
import requests
import soundfile as sf
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel


# ============================================================
# CONFIG
# ============================================================

HOST = os.environ.get("KOKORO_HOST", "0.0.0.0")
PORT = int(os.environ.get("KOKORO_PORT", "8003"))  # the port agent_api calls

# Kokoro voice ids. Hindi voices use the hf_/hm_ prefix. Override via env if
# a different voice suits Divya better -- /voices lists what actually loaded.
KOKORO_HI_VOICE = os.environ.get("KOKORO_HI_VOICE", "hf_alpha")
KOKORO_EN_VOICE = os.environ.get("KOKORO_EN_VOICE", "hf_alpha")
KOKORO_SPEED = float(os.environ.get("KOKORO_SPEED", "1.0"))

# Svara adapter, used for Kannada and as a safety net on any Kokoro error.
SVARA_FALLBACK_URL = os.environ.get(
    "SVARA_FALLBACK_URL", "http://127.0.0.1:8007/synthesize"
)

SAMPLE_RATE = 24000  # Kokoro outputs 24kHz

# Unicode ranges used for script-based routing.
_DEVANAGARI = (0x0900, 0x097F)   # Hindi, Marathi
_KANNADA = (0x0C80, 0x0CFF)

http_session = requests.Session()
http_session.mount(
    "http://", requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20)
)

app = FastAPI(title="Kokoro TTS (Svara-compatible)")

_pipeline = None
_pipeline_lock = threading.Lock()  # KPipeline is not documented as thread-safe
_load_error = None


class TTSRequest(BaseModel):
    text: str
    language: str = None
    voice: str = None


# ============================================================
# MODEL
# ============================================================

def load_pipeline():
    """Load Kokoro once. lang_code 'h' is Hindi; it still renders the Latin
    and phonetic-English fragments that show up in Hinglish replies."""
    global _pipeline, _load_error
    if _pipeline is not None or _load_error is not None:
        return _pipeline

    try:
        from kokoro import KPipeline
    except ImportError as e:
        _load_error = (
            f"kokoro package not installed ({e}). pip install kokoro soundfile"
        )
        print("KOKORO LOAD ERROR:", _load_error)
        return None

    print("=" * 60)
    print("Loading Kokoro-82M (lang_code='h')")
    print("=" * 60)
    t0 = time.perf_counter()
    try:
        _pipeline = KPipeline(lang_code="h")
    except Exception as e:
        _load_error = f"KPipeline init failed: {e}"
        print("KOKORO LOAD ERROR:", _load_error)
        return None
    print(f"Kokoro loaded in {time.perf_counter() - t0:.1f}s")
    return _pipeline


def _synthesize_kokoro(text: str, voice: str) -> np.ndarray:
    """Run Kokoro and return one float32 mono waveform at 24kHz.

    KPipeline yields (graphemes, phonemes, audio) per segment; long text can
    come back as several segments, so concatenate them.
    """
    pipeline = load_pipeline()
    if pipeline is None:
        raise RuntimeError(_load_error or "Kokoro pipeline unavailable")

    segments = []
    with _pipeline_lock:
        for item in pipeline(text, voice=voice, speed=KOKORO_SPEED):
            # Tolerate both the 3-tuple form and a bare-audio form.
            audio = item[2] if isinstance(item, (tuple, list)) else item
            if hasattr(audio, "detach"):        # torch tensor
                audio = audio.detach().cpu().numpy()
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio.size:
                segments.append(audio)

    if not segments:
        raise RuntimeError("Kokoro returned no audio")
    return np.concatenate(segments) if len(segments) > 1 else segments[0]


def _wav_b64(audio: np.ndarray) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================
# ROUTING
# ============================================================

def _has_script(text: str, lo: int, hi: int) -> bool:
    return any(lo <= ord(ch) <= hi for ch in text)


def _route(text: str, language: str) -> str:
    """Return 'svara' or 'kokoro'."""
    lang = (language or "").lower()
    if lang in ("kn", "kannada"):
        return "svara"
    if _has_script(text, *_KANNADA):
        return "svara"
    return "kokoro"


def _pick_voice(text: str, explicit: str) -> str:
    if explicit:
        return explicit
    if _has_script(text, *_DEVANAGARI):
        return KOKORO_HI_VOICE
    return KOKORO_EN_VOICE


def _svara_fallback(text: str, language: str, reason: str):
    """Forward to the Svara adapter, which speaks this same contract."""
    print(f"[TTS] falling back to Svara ({reason})")
    payload = {"text": text}
    if language:
        payload["language"] = language
    resp = http_session.post(SVARA_FALLBACK_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


# ============================================================
# ROUTES
# ============================================================

@app.on_event("startup")
def startup_event():
    # Load and warm up now, so the first real caller doesn't pay for it.
    if load_pipeline() is not None:
        try:
            t0 = time.perf_counter()
            _synthesize_kokoro("नमस्ते", KOKORO_HI_VOICE)
            print(f"Kokoro warmup synth: {time.perf_counter() - t0:.3f}s")
        except Exception as e:
            print("Kokoro warmup failed (will still try per-request):", e)


@app.get("/health")
def health():
    return {
        "status": "ok" if _pipeline is not None else "degraded",
        "service": "kokoro-tts",
        "kokoro_loaded": _pipeline is not None,
        "load_error": _load_error,
        "hi_voice": KOKORO_HI_VOICE,
        "sample_rate": SAMPLE_RATE,
        "svara_fallback": SVARA_FALLBACK_URL,
    }


@app.post("/synthesize")
def synthesize(request: TTSRequest):
    text = (request.text or "").strip()
    if not text:
        return {"status": "error", "message": "Text cannot be empty"}

    backend = _route(text, request.language)

    if backend == "kokoro":
        voice = _pick_voice(text, request.voice)
        t0 = time.perf_counter()
        try:
            audio = _synthesize_kokoro(text, voice)
            elapsed = time.perf_counter() - t0
            dur = len(audio) / SAMPLE_RATE
            print(
                f"[TTS] kokoro {elapsed:.3f}s for {dur:.2f}s audio "
                f"(RTF {elapsed / dur:.3f}) voice={voice}"
            )
            return {
                "status": "success",
                "text": text,
                "voice": voice,
                "engine": "kokoro",
                "audio_b64": _wav_b64(audio),
                "sample_rate": SAMPLE_RATE,
            }
        except Exception as e:
            print("KOKORO TTS ERROR:", e)
            backend = "svara"  # fall through rather than fail the turn

    try:
        result = _svara_fallback(text, request.language, reason=backend)
        result.setdefault("engine", "svara")
        return result
    except Exception as e:
        print("SVARA FALLBACK ERROR:", e)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "stage": "tts", "error": str(e)},
        )


@app.get("/voices")
def voices():
    """Best-effort listing so the real Hindi voice ids can be confirmed
    rather than guessed."""
    pipeline = load_pipeline()
    if pipeline is None:
        return {"status": "error", "error": _load_error}
    found = []
    for attr in ("voices", "available_voices"):
        val = getattr(pipeline, attr, None)
        if val:
            try:
                found = sorted(val.keys()) if hasattr(val, "keys") else sorted(val)
            except Exception:
                found = [str(val)]
            break
    return {"status": "ok", "voices": found, "configured_hi": KOKORO_HI_VOICE}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
