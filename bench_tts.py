"""
A/B latency benchmark: Kokoro (8003) vs Svara (8007).

Both speak the same /synthesize contract, so this sends identical sentences
to each and reports per-clause latency plus real-time factor. These are real
Divya replies pulled from call logs, including the short first-clause chunks
that actually determine how fast the call feels.

    python bench_tts.py

TTS_FIRST_CHUNK measured on live calls with Svara ranged 1.3-8.7s. The first
clause is the one that matters -- it is what the caller waits for.
"""

import base64
import io
import statistics
import sys
import time

import requests

KOKORO_URL = "http://127.0.0.1:8003/synthesize"
SVARA_URL = "http://127.0.0.1:8007/synthesize"

# Real replies from the call logs. First entry is a short opening clause,
# which is the latency that the caller actually perceives.
SENTENCES = [
    "जी, बताइए",
    "हमारा एमबीए प्रोग्राम दो साल का फुल-टाइम कोर्स है।",
    "हमारे पास मार्केटिंग, फाइनेंस, ह्यूमन रिसोर्स मैनेजमेंट, और बिज़नेस एनालिटिक्स में स्पेशलाइज़ेशन उपलब्ध हैं।",
    "प्लेसमेंट सेल पूरे साल काम करता है और तैयारी पहले सेमेस्टर से ही शुरू हो जाती है।",
]


def audio_seconds(audio_b64, sample_rate):
    """Decode just enough to get duration, for the real-time factor."""
    try:
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(base64.b64decode(audio_b64)))
        return len(data) / sr
    except Exception:
        # Rough fallback: 16-bit mono PCM with a 44-byte header
        return max(len(base64.b64decode(audio_b64)) - 44, 0) / (2 * sample_rate)


def bench(name, url, sentences):
    print(f"\n{'=' * 62}\n{name}  ({url})\n{'=' * 62}")
    times = []
    for s in sentences:
        t0 = time.perf_counter()
        try:
            r = requests.post(url, json={"text": s}, timeout=180)
            r.raise_for_status()
            j = r.json()
        except Exception as e:
            print(f"  ERROR on {s[:32]!r}: {e}")
            continue
        elapsed = time.perf_counter() - t0

        b64 = j.get("audio_b64")
        if not b64:
            print(f"  NO AUDIO for {s[:32]!r}: {j}")
            continue

        dur = audio_seconds(b64, j.get("sample_rate", 24000))
        rtf = elapsed / dur if dur else float("nan")
        engine = j.get("engine", "?")
        times.append(elapsed)
        print(
            f"  {elapsed:6.3f}s  (audio {dur:5.2f}s, RTF {rtf:5.3f}, "
            f"engine={engine})  {s[:40]}"
        )

    if times:
        print(f"\n  first clause : {times[0]:.3f}s   <-- what the caller waits for")
        print(f"  median       : {statistics.median(times):.3f}s")
        print(f"  worst        : {max(times):.3f}s")
    return times


def main():
    # Warm both so we compare steady-state, not one-off model load.
    for url in (KOKORO_URL, SVARA_URL):
        try:
            requests.post(url, json={"text": "नमस्ते"}, timeout=180)
        except Exception:
            pass

    k = bench("KOKORO", KOKORO_URL, SENTENCES)
    s = bench("SVARA", SVARA_URL, SENTENCES)

    print(f"\n{'=' * 62}\nVERDICT\n{'=' * 62}")
    if k and s:
        kf, sf_ = k[0], s[0]
        print(f"  first-clause latency : Kokoro {kf:.3f}s  vs  Svara {sf_:.3f}s")
        if kf > 0:
            print(f"  speedup on first clause: {sf_ / kf:.1f}x")
        km, sm = statistics.median(k), statistics.median(s)
        print(f"  median               : Kokoro {km:.3f}s  vs  Svara {sm:.3f}s")
        print()
        print("  Live calls showed FIRST_AUDIO = GEMMA_FIRST_SENTENCE + TTS_FIRST_CHUNK.")
        print(f"  Replacing ~{sm:.1f}s of TTS with ~{km:.1f}s removes that from every turn,")
        print("  and frees the GPU Gemma was competing with.")
    else:
        print("  Could not benchmark both engines -- check that each is running.")
        if not k:
            print(f"  Kokoro unreachable at {KOKORO_URL}")
        if not s:
            print(f"  Svara unreachable at {SVARA_URL}")
        sys.exit(1)


if __name__ == "__main__":
    main()
