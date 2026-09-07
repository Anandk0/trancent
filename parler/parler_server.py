import io
import torch
import soundfile as sf

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import Response

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer


MODEL_ID = "ai4bharat/indic-parler-tts"
DEVICE = "cuda"

app = FastAPI(title="Indic Parler TTS")


print("Loading Indic Parler...")
model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_ID
).to(DEVICE)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)

print("Indic Parler loaded.")
print("Sampling rate:", model.config.sampling_rate)


class TTSRequest(BaseModel):
    text: str
    description: str = (
        "Rani is a young female Hindi speaker. "
        "She has a warm, friendly and pleasant voice. "
        "She speaks clearly and naturally."
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "sampling_rate": model.config.sampling_rate,
    }


@app.post("/tts")
def tts(request: TTSRequest):

    description_inputs = description_tokenizer(
        request.description,
        return_tensors="pt"
    ).to(DEVICE)

    prompt_inputs = tokenizer(
        request.text,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.inference_mode():

        generation = model.generate(
            input_ids=description_inputs.input_ids,
            attention_mask=description_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
        )

    audio = generation.cpu().numpy().squeeze()

    buffer = io.BytesIO()

    sf.write(
        buffer,
        audio,
        model.config.sampling_rate,
        format="WAV"
    )

    buffer.seek(0)

    return Response(
        content=buffer.read(),
        media_type="audio/wav"
    )
