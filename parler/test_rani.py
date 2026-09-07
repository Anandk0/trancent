import torch
import soundfile as sf

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

MODEL_ID = "ai4bharat/indic-parler-tts"
DEVICE = "cuda"

print("Loading model...")

model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_ID
).to(DEVICE)

# IMPORTANT: separate tokenizers
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)

description = (
    "Rani is a young female Hindi speaker. "
    "She has a warm, friendly and pleasant voice. "
    "She speaks clearly and naturally."
)

text = (
    "नमस्ते मैडम... मैं रानी जैन हेरिटेज स्कूल बेलगावी से बोल रही हूँ.... "
    "हमारे स्कूल में नए शैक्षणिक सत्र के लिए.... एडमिशन शुरू हो गए हैं.....। "
    "मैं आपको स्कूल के बारे में.....  पूरी जानकारी दे सकती हूँ।"
)

print("Description:", description)
print("Text:", text)

description_inputs = description_tokenizer(
    description,
    return_tensors="pt"
).to(DEVICE)

prompt_inputs = tokenizer(
    text,
    return_tensors="pt"
).to(DEVICE)

print("Generating...")

with torch.no_grad():
    generation = model.generate(
        input_ids=description_inputs.input_ids,
        attention_mask=description_inputs.attention_mask,
        prompt_input_ids=prompt_inputs.input_ids,
        prompt_attention_mask=prompt_inputs.attention_mask,
    )

audio = generation.cpu().numpy().squeeze()

output = "rani_baseline.wav"

sf.write(
    output,
    audio,
    model.config.sampling_rate
)

duration = len(audio) / model.config.sampling_rate

print()
print("================================")
print("DONE")
print("File:", output)
print("Sample rate:", model.config.sampling_rate)
print("Duration:", round(duration, 2), "seconds")
print("================================")
