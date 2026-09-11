# Kokoro TTS — setup and cutover

## Why

Live-call measurements showed `FIRST_AUDIO_READY == GEMMA_FIRST_SENTENCE + TTS_FIRST_CHUNK`
exactly, every turn, with the sum pinned around 10–13s while the split between the two swung
wildly:

| Turn | Gemma first sentence | TTS first chunk | First audio |
|------|---------------------|-----------------|-------------|
| 1:15 | 10.87s              | 2.63s           | 13.50s      |
| 1:18 | **2.31s**           | **7.80s**       | 10.11s      |
| 1:19 | **2.51s**           | **7.74s**       | 10.25s      |

Constant total, shifting split — Gemma (4B) and Svara (3B) are fighting over the same MIG slice.

Svara is Orpheus-style **autoregressive**: it emits ~85 audio tokens per second of speech, one at
a time, on that contended GPU. Kokoro-82M is StyleTTS2-based and **non-autoregressive** — one
forward pass for the whole waveform, RTF ~0.03, so a 3-second clause takes ~100ms.

That is why this swap wins twice: TTS gets ~50x faster **and** an 82M model barely touches the
GPU, so Gemma stops competing and gets its own throughput back.

## Install

In whichever env will run the TTS server (a fresh env is fine — Kokoro has no heavy deps):

```bash
pip install kokoro soundfile
```

Kokoro needs `espeak-ng` for phonemisation:

```bash
conda install -c conda-forge -y espeak-ng
# or: apt-get install -y espeak-ng
```

## Port layout

| Service | Port | Notes |
|---------|------|-------|
| `kokoro_server.py` | **8003** | What `agent_api.py` already calls — no agent change needed |
| `svara_tts_api.py` | **8007** | Moved. Now the Kannada route + fallback |
| Svara model server | 8095 | Unchanged |

## Cutover

```bash
cd ~/newollama-volume
git pull origin main

# 1. Move the Svara adapter off 8003 (it becomes the fallback)
pkill -f svara_tts_api.py
sleep 1
python svara_tts_api.py &        # now binds 8007 by default

# 2. Start Kokoro on 8003
python kokoro_server.py          # loads the model, warms up, then serves
```

Verify both:

```bash
curl -s http://127.0.0.1:8003/health   # kokoro_loaded should be true
curl -s http://127.0.0.1:8007/health   # svara adapter
```

If `kokoro_loaded` is `false`, `load_error` in that response says why.

## Prove the win

```bash
python bench_tts.py
```

Sends identical real Divya sentences to both engines and reports first-clause latency, median,
and real-time factor. The **first clause** number is the one that matters — it is what the caller
waits through.

## Confirm the voice

Voice ids are configurable rather than hardcoded, because the exact Hindi ids should be confirmed
rather than guessed:

```bash
curl -s http://127.0.0.1:8003/voices
```

To change:

```bash
KOKORO_HI_VOICE=hf_beta python kokoro_server.py
```

Other env knobs: `KOKORO_EN_VOICE`, `KOKORO_SPEED`, `KOKORO_PORT`, `SVARA_FALLBACK_URL`.

## Safety

This cannot regress below today's behaviour:

- Kannada text routes to Svara automatically (Kokoro covers Hindi + English, not Kannada/Marathi).
- **Any** Kokoro error falls through to Svara for that request.
- To revert entirely: stop `kokoro_server.py` and run `SVARA_ADAPTER_PORT=8003 python svara_tts_api.py`.

## What's next

This is Phase 1. Remaining latency after it lands:

1. **Gemma TTFT** — the system prompt is re-prefilled every turn (measured 1.92s to first token
   on an idle GPU). vLLM with automatic prefix caching takes this to ~0.05s. Biggest remaining win.
2. **`SILENCE_MS` = 1.6s** — dead time before the pipeline even starts, so it is the largest single
   block once 1 lands. Needs smarter turn detection than a fixed timer.
