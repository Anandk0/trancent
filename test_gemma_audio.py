"""
Feed Gemma an audio file directly and print what it says back.

Gemma4Processor exposes an audio tower (_process_audio, audio_token,
audio_ms_per_token), so the caller's speech may be able to go straight to
the LLM instead of through a separate ASR stage -- which is where two
thirds of a turn currently goes.

This is a throwaway probe, not a service. It loads its own copy of the
model (~8GB on top of whatever is already resident) and exits.

    python test_gemma_audio.py /tmp/asr_test.wav
    python test_gemma_audio.py /tmp/asr_test.wav --mode transcribe

The exact content-dict shape for audio is not documented the same way
across processors, so several are tried in turn and the first that builds
is reported -- that shape is the thing to copy into gemma_server.py if this
works.
"""

import argparse
import sys
import time

import torch

MODEL_ID = "google/gemma-4-E4B-it"

PROMPTS = {
    "reply": (
        "You are Divya, an admission counsellor for Jain College of "
        "Engineering and Research, Belagavi. The caller just said the "
        "attached audio. Reply naturally in the same language they used, "
        "in one or two sentences."
    ),
    "transcribe": "Transcribe the attached audio exactly. Output only the transcript.",
}


def load_audio(path, target_sr=16000):
    """Return (samples, sample_rate) as float32 mono at target_sr."""
    import soundfile as sf
    import numpy as np

    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        try:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            # Crude decimation beats failing outright for a probe; the model
            # may still understand it well enough to tell us whether the
            # audio path works at all.
            step = sr / target_sr
            idx = (np.arange(int(len(data) / step)) * step).astype(int)
            data = data[idx]
            print(f"  (librosa missing - crude resample {sr} -> {target_sr})")
        sr = target_sr
    return data, sr


def candidate_contents(path, samples, sr):
    """Content dicts to try, most likely first."""
    return [
        ("audio=array", {"type": "audio", "audio": samples}),
        ("audio=path", {"type": "audio", "audio": path}),
        ("path=", {"type": "audio", "path": path}),
        ("url=", {"type": "audio", "url": path}),
        ("audio=(array,sr)", {"type": "audio", "audio": (samples, sr)}),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="path to a wav file")
    ap.add_argument("--mode", choices=list(PROMPTS), default="reply")
    ap.add_argument("--max-new-tokens", type=int, default=72)
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForMultimodalLM

    print("loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"  audio_token       = {getattr(processor, 'audio_token', '?')}")
    print(f"  audio_ms_per_token= {getattr(processor, 'audio_ms_per_token', '?')}")
    print(f"  audio_seq_length  = {getattr(processor, 'audio_seq_length', '?')}")

    print(f"loading audio {args.audio} ...")
    samples, sr = load_audio(args.audio)
    print(f"  {len(samples)} samples @ {sr} Hz = {len(samples)/sr:.2f}s")

    print("loading model (this is the slow part)...")
    t0 = time.perf_counter()
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    inputs = None
    used = None
    for name, content in candidate_contents(args.audio, samples, sr):
        messages = [{
            "role": "user",
            "content": [content, {"type": "text", "text": PROMPTS[args.mode]}],
        }]
        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
            used = name
            print(f"\naccepted content shape: {name}")
            break
        except Exception as e:
            print(f"  {name:<18} rejected: {type(e).__name__}: {str(e)[:110]}")

    if inputs is None:
        print("\nNo content shape was accepted. The processor's audio path needs")
        print("a different call than any tried here.")
        sys.exit(1)

    dev = next(model.parameters()).device
    inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}

    n_in = inputs["input_ids"].shape[-1]
    print(f"prompt tokens: {n_in}  (audio becomes part of this)")

    print("\ngenerating...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    elapsed = time.perf_counter() - t0

    reply = processor.decode(out[0][n_in:], skip_special_tokens=True)

    print("=" * 60)
    print(reply.strip())
    print("=" * 60)
    print(f"{elapsed:.2f}s for {out.shape[-1] - n_in} tokens "
          f"({(out.shape[-1] - n_in) / elapsed:.1f} tok/s)")
    print(f"content shape that worked: {used}")


if __name__ == "__main__":
    main()
