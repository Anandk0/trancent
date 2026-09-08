import io
import base64
import torch
import soundfile as sf

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import uvicorn


MODEL_ID = "ai4bharat/indic-parler-tts"
HOST = "0.0.0.0"
PORT = 8003
OUTPUT_PATH = "/tmp/tts_output.wav"

DIVYA_DESCRIPTION = (
    "Divya's voice is monotone yet slightly fast in delivery, "
    "with a very close recording that almost has no background noise."
)

app = FastAPI(title="Indic Parler TTS")

model = None
tokenizer = None
description_tokenizer = None
cached_description_inputs = None


class TTSRequest(BaseModel):
    text: str


def load_model():
    global model
    global tokenizer
    global description_tokenizer
    global cached_description_inputs

    print("Loading Indic Parler TTS...")

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    # IMPORTANT:
    # Indic Parler TTS uses a separate tokenizer for descriptions.
    description_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    model.eval()

    # Pre-tokenize Divya description once at startup to avoid repeated tokenization overhead
    print("Pre-caching Divya description tokens...")
    desc_inputs = description_tokenizer(
        DIVYA_DESCRIPTION,
        return_tensors="pt"
    )
    if torch.cuda.is_available():
        desc_inputs = {k: v.to("cuda") for k, v in desc_inputs.items()}
    cached_description_inputs = desc_inputs

    print("Model and description cache ready.")
    print("Description tokenizer:", model.config.text_encoder._name_or_path)


@app.on_event("startup")
def startup():
    load_model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "cuda": torch.cuda.is_available()
    }


@app.post("/synthesize")
def synthesize(request: TTSRequest):
    global cached_description_inputs

    text = request.text.strip()

    if not text:
        return {
            "status": "error",
            "message": "Text cannot be empty"
        }

    print("\n" + "=" * 40)
    print("TTS TEXT:", text)
    print("=" * 40)

    # Ensure cached description inputs are loaded
    if cached_description_inputs is None:
        desc_inputs = description_tokenizer(
            DIVYA_DESCRIPTION,
            return_tensors="pt"
        )
        if torch.cuda.is_available():
            desc_inputs = {k: v.to("cuda") for k, v in desc_inputs.items()}
        cached_description_inputs = desc_inputs

    # Tokenize user text prompt
    prompt_inputs = tokenizer(
        text,
        return_tensors="pt"
    )
    if torch.cuda.is_available():
        prompt_inputs = {k: v.to("cuda") for k, v in prompt_inputs.items()}

    with torch.inference_mode():
        generation = model.generate(
            input_ids=cached_description_inputs["input_ids"],
            attention_mask=cached_description_inputs["attention_mask"],
            prompt_input_ids=prompt_inputs["input_ids"],
            prompt_attention_mask=prompt_inputs["attention_mask"]
        )

    audio = generation.cpu().float().numpy().squeeze()
    sample_rate = model.config.sampling_rate

    # Generate in-memory WAV and base64
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    audio_bytes = buffer.getvalue()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    # Also write to OUTPUT_PATH for backward compatibility
    try:
        with open(OUTPUT_PATH, "wb") as f:
            f.write(audio_bytes)
    except Exception as e:
        print("[WARN] Could not write temp WAV file:", e)

    print("Synthesized audio bytes:", len(audio_bytes), "sample_rate:", sample_rate)

    return {
        "status": "success",
        "text": text,
        "audio_file": OUTPUT_PATH,
        "audio_b64": audio_b64,
        "sample_rate": sample_rate
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT
    )