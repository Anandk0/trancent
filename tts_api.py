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

app = FastAPI(title="Indic Parler TTS")


model = None
tokenizer = None
description_tokenizer = None


class TTSRequest(BaseModel):
    text: str


def load_model():
    global model
    global tokenizer
    global description_tokenizer

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

    if torch.cuda.is_available():
        model = model.to("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    model.eval()

    print("Model loaded")
    print("Description tokenizer:",
          model.config.text_encoder._name_or_path)


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

    text = request.text.strip()

    if not text:
        return {
            "status": "error",
            "message": "Text cannot be empty"
        }

    # Keep this extremely simple for the first test.
    # This is the official Divya speaker conditioning.
    description = (
        "Divya's voice is monotone yet slightly fast in delivery, "
        "with a very close recording that almost has no background noise."
    )

    print("\n==============================")
    print("TEXT:")
    print(text)
    print("\nDESCRIPTION:")
    print(description)
    print("==============================")

    # Description tokenizer
    description_inputs = description_tokenizer(
        description,
        return_tensors="pt"
    )

    # Speech/text tokenizer
    prompt_inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    if torch.cuda.is_available():

        description_inputs = {
            key: value.to("cuda")
            for key, value in description_inputs.items()
        }

        prompt_inputs = {
            key: value.to("cuda")
            for key, value in prompt_inputs.items()
        }

    print("Generating...")

    with torch.no_grad():

        generation = model.generate(
            input_ids=description_inputs["input_ids"],
            attention_mask=description_inputs["attention_mask"],
            prompt_input_ids=prompt_inputs["input_ids"],
            prompt_attention_mask=prompt_inputs["attention_mask"]
        )

    audio = generation.cpu().float().numpy().squeeze()

    sf.write(
        OUTPUT_PATH,
        audio,
        model.config.sampling_rate
    )

    print("Generated:", OUTPUT_PATH)
    print("Sample rate:", model.config.sampling_rate)
    print("Audio samples:", len(audio))

    return {
        "status": "success",
        "text": text,
        "audio_file": OUTPUT_PATH,
        "sample_rate": model.config.sampling_rate
    }


if __name__ == "__main__":

    uvicorn.run(
        app,
        host=HOST,
        port=PORT
    )