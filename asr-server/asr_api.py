import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
import subprocess
import numpy as np
import torch
import torchaudio

from fastapi import FastAPI, UploadFile, File, Form
from transformers import AutoModel


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
TARGET_SAMPLE_RATE = 16000
DEFAULT_LANGUAGE = "hi"

# Optimize CPU threads for PyTorch/ONNX inference
num_cpus = os.cpu_count() or 4
torch.set_num_threads(min(num_cpus, 8))


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Indic Conformer ASR API",
    version="1.0",
)

asr_model = None


# ============================================================
# LANGUAGE CODES
# ============================================================

SUPPORTED_LANGUAGES = {
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "kok": "Konkani",
    "ks": "Kashmiri",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}


# ============================================================
# LOAD MODEL
# ============================================================

def load_asr():
    global asr_model

    if asr_model is not None:
        return asr_model

    print("=" * 60)
    print("Loading Indic Conformer (CPU)")
    print("=" * 60)
    print("Model:", MODEL_ID)

    asr_model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    asr_model.eval()

    print("Indic Conformer loaded successfully.")
    return asr_model


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    load_asr()


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "indic-conformer-asr",
        "model": MODEL_ID,
        "cuda": torch.cuda.is_available(),
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "service": "Indic Conformer ASR",
        "status": "running",
        "endpoint": "/transcribe",
        "languages": SUPPORTED_LANGUAGES,
    }


# ============================================================
# TRANSCRIBE
# ============================================================

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(DEFAULT_LANGUAGE),
    decoding: str = Form("ctc"),
):
    t_start = time.perf_counter()

    print()
    print("=" * 60)
    print("ASR REQUEST")
    print("Filename:", file.filename)
    print("Language:", language)
    print("Decoding:", decoding)
    print("=" * 60)

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------
    if language not in SUPPORTED_LANGUAGES:
        return {
            "status": "error",
            "error": f"Unsupported language: {language}",
            "supported_languages": SUPPORTED_LANGUAGES,
        }

    if decoding not in ["ctc", "rnnt"]:
        return {
            "status": "error",
            "error": "decoding must be either 'ctc' or 'rnnt'",
        }

    # --------------------------------------------------------
    # Direct In-Memory Read and FFmpeg Pipe Decode
    # --------------------------------------------------------
    data = await file.read()
    if not data:
        return {
            "status": "error",
            "error": "Empty audio data received"
        }

    t_decode_start = time.perf_counter()
    try:
        # Decode directly from memory via pipe:0 (avoids disk tempfile write & read)
        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-ar", str(TARGET_SAMPLE_RATE),
            "-ac", "1",
            "-f", "f32le",
            "-"
        ]

        result = subprocess.run(
            ffmpeg_command,
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        audio_np = np.frombuffer(
            result.stdout,
            dtype=np.float32,
        )

        wav = torch.from_numpy(audio_np).unsqueeze(0)
        t_decode_elapsed = time.perf_counter() - t_decode_start

        # ----------------------------------------------------
        # Run ASR Inference
        # ----------------------------------------------------
        model = load_asr()
        t_infer_start = time.perf_counter()

        with torch.inference_mode():
            transcription = model(
                wav,
                language,
                decoding,
            )

        t_infer_elapsed = time.perf_counter() - t_infer_start

        if isinstance(transcription, (list, tuple)):
            transcription = transcription[0]
        transcription = str(transcription).strip()

        t_total = time.perf_counter() - t_start

        print(f"[TIMING] ASR_DECODE: {t_decode_elapsed:.3f}s | ASR_INFER: {t_infer_elapsed:.3f}s | TOTAL: {t_total:.3f}s")
        print(f"TRANSCRIPTION: '{transcription}'")
        print("=" * 60)

        return {
            "status": "success",
            "filename": file.filename,
            "language": language,
            "language_name": SUPPORTED_LANGUAGES[language],
            "decoding": decoding,
            "sample_rate": TARGET_SAMPLE_RATE,
            "text": transcription,
            "timings": {
                "decode_s": round(t_decode_elapsed, 3),
                "infer_s": round(t_infer_elapsed, 3),
                "total_s": round(t_total, 3)
            }
        }

    except Exception as e:
        print("=" * 60)
        print("ASR ERROR:", type(e).__name__, str(e))
        print("=" * 60)

        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e),
        }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
    )