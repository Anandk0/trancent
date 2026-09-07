import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tempfile

import torch
import torchaudio

from fastapi import FastAPI, UploadFile, File, Form
from transformers import AutoModel


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

TARGET_SAMPLE_RATE = 16000

# Default language
DEFAULT_LANGUAGE = "hi"

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
    print("Loading Indic Conformer")
    print("=" * 60)

    print("Model:", MODEL_ID)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    asr_model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

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

    print()
    print("=" * 60)
    print("ASR REQUEST")
    print("Filename:", file.filename)
    print("Content type:", file.content_type)
    print("Language:", language)
    print("Decoding:", decoding)
    print("=" * 60)

    # --------------------------------------------------------
    # Validate language
    # --------------------------------------------------------

    if language not in SUPPORTED_LANGUAGES:

        return {
            "status": "error",
            "error": f"Unsupported language: {language}",
            "supported_languages": SUPPORTED_LANGUAGES,
        }

    # --------------------------------------------------------
    # Validate decoding
    # --------------------------------------------------------

    if decoding not in ["ctc", "rnnt"]:

        return {
            "status": "error",
            "error": "decoding must be either 'ctc' or 'rnnt'",
        }

    # --------------------------------------------------------
    # Save uploaded file
    # --------------------------------------------------------

    suffix = os.path.splitext(file.filename or ".wav")[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as tmp:

        data = await file.read()

        tmp.write(data)

        audio_path = tmp.name

    print("Audio saved:", audio_path)

    try:

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model = load_asr()

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        print("Loading audio...")

        # ----------------------------------------------------
        # Decode and resample audio using FFmpeg
        # ----------------------------------------------------

        print("Decoding audio with FFmpeg...")

        import subprocess
        import numpy as np

        ffmpeg_command = [
            "ffmpeg",
            "-i", audio_path,
            "-ar", str(TARGET_SAMPLE_RATE),
            "-ac", "1",
            "-f", "f32le",
            "-"
        ]

        result = subprocess.run(
            ffmpeg_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        # FFmpeg returns raw float32 PCM
        audio_np = np.frombuffer(
            result.stdout,
            dtype=np.float32,
        )

        # Convert to PyTorch tensor
        wav = torch.from_numpy(audio_np).unsqueeze(0)

        print("Final audio shape:", tuple(wav.shape))
        print("Final sample rate:", TARGET_SAMPLE_RATE)

        print("Final audio shape:", tuple(wav.shape))
        print("Final sample rate:", TARGET_SAMPLE_RATE)

        # ----------------------------------------------------
        # Move tensor to correct device if necessary
        # ----------------------------------------------------

        # The Indic Conformer ONNX implementation handles
        # its own execution providers.
        #
        # Keep audio tensor on CPU.

        # ----------------------------------------------------
        # RUN ASR
        # ----------------------------------------------------

        print("Running ASR...")

        with torch.no_grad():

            transcription = model(
                wav,
                language,
                decoding,
            )

        # ----------------------------------------------------
        # Normalize output
        # ----------------------------------------------------

        if isinstance(transcription, (list, tuple)):

            transcription = transcription[0]

        transcription = str(transcription)

        print("TRANSCRIPTION:")
        print(transcription)

        print("=" * 60)

        return {
            "status": "success",
            "filename": file.filename,
            "language": language,
            "language_name": SUPPORTED_LANGUAGES[language],
            "decoding": decoding,
            "sample_rate": TARGET_SAMPLE_RATE,
            "text": transcription,
        }

    except Exception as e:

        print()
        print("=" * 60)
        print("ASR ERROR")
        print(type(e).__name__)
        print(str(e))
        print("=" * 60)

        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e),
        }

    finally:

        # ----------------------------------------------------
        # Delete temporary file
        # ----------------------------------------------------

        try:
            os.remove(audio_path)
        except Exception:
            pass


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