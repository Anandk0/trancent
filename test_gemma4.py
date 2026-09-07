import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-E4B-it"

print("=" * 60)
print("Gemma 4 E4B test")
print("=" * 60)

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    "GB",
)

print("\nLoading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("Processor loaded.")

print("\nLoading Gemma with CPU offload...")

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    offload_buffers=True,
)

print("\n" + "=" * 60)
print("MODEL LOADED SUCCESSFULLY")
print("=" * 60)

print("Model type:", type(model))

print("\nTesting generation...")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Haan, hum agle somvaar ko school visit karna chahenge. Iska short reply Hindi Hinglish mein do."
            }
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
)

# Put inputs on the model's first execution device.
input_device = next(model.parameters()).device
inputs = {
    k: v.to(input_device) if hasattr(v, "to") else v
    for k, v in inputs.items()
}

print("Input device:", input_device)
print("Generating...")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

input_length = inputs["input_ids"].shape[-1]

response = processor.decode(
    output[0][input_length:],
    skip_special_tokens=True,
)

print("\n" + "=" * 60)
print("GEMMA RESPONSE:")
print(response)
print("=" * 60)

print(
    "\nGPU allocated:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB",
)
