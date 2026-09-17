"""
A/B latency benchmark between two TTS endpoints speaking the same contract.

Both sides are just a /synthesize URL, so this compares whatever two engines
or placements you point it at -- local Svara vs Svara on another MIG slice,
or Svara vs a different engine entirely.

    python bench_tts.py                          # 8003 vs 8007
    BENCH_B_URL=http://10.233.119.239:8003/synthesize python bench_tts.py

Environment:
    BENCH_A_URL / BENCH_A_NAME     default http://127.0.0.1:8003  "A (8003)"
    BENCH_B_URL / BENCH_B_NAME     default http://127.0.0.1:8007  "B (8007)"

The FIRST CLAUSE number is the one that matters -- it is what the caller
waits through in silence. Median matters for the rest of the reply, which is
spoken while earlier clauses are already playing.
"""

import base64
import io
import os
import statistics
import sys
import time

import requests

A_URL = os.environ.get("BENCH_A_URL", "http://127.0.0.1:8003/synthesize")
B_URL = os.environ.get("BENCH_B_URL", "http://127.0.0.1:8007/synthesize")
A_NAME = os.environ.get("BENCH_A_NAME", "A  (8003)")
B_NAME = os.environ.get("BENCH_B_NAME", "B  (8007)")

# Real replies from the call logs. The first is a short opening clause --
# the same shape extract_first_chunk() emits, and the latency the caller
# actually perceives.
SENTENCES = [
    "जी, बताइए",
    "हमारा एमबीए प्रोग्राम दो साल का फुल-टाइम कोर्स है।",
    "हमारे पास मार्केटिंग, फाइनेंस, ह्यूमन रिसोर्स मैनेजमेंट, और बिज़नेस एनालिटिक्स में स्पेशलाइज़ेशन उपलब्ध हैं।",
    "प्लेसमेंट सेल पूरे साल काम करता है और तैयारी पहले सेमेस्टर से ही शुरू हो जाती है।",
]

REPEATS = int(os.environ.get("BENCH_REPEATS", "1"))


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
    print(f"\n{'=' * 66}\n{name}   {url}\n{'=' * 66}")
    times = []
    for rep in range(REPEATS):
        for s in sentences:
            t0 = time.perf_counter()
            try:
                r = requests.post(url, json={"text": s}, timeout=180)
                r.raise_for_status()
                j = r.json()
            except Exception as e:
                print(f"  ERROR on {s[:30]!r}: {type(e).__name__}: {e}")
                continue
            elapsed = time.perf_counter() - t0

            b64 = j.get("audio_b64")
            if not b64:
                print(f"  NO AUDIO for {s[:30]!r}: {str(j)[:120]}")
                continue

            dur = audio_seconds(b64, j.get("sample_rate", 24000))
            rtf = elapsed / dur if dur else float("nan")
            times.append(elapsed)
            print(
                f"  {elapsed:6.3f}s   audio {dur:5.2f}s   RTF {rtf:5.3f}   {s[:38]}"
            )

    if times:
        print(f"\n  first clause : {times[0]:6.3f}s   <- what the caller waits for")
        print(f"  median       : {statistics.median(times):6.3f}s")
        print(f"  worst        : {max(times):6.3f}s")
    return times


def main():
    # Warm both so we compare steady state rather than a cold first call --
    # vLLM's first request after startup pays for CUDA graph capture and
    # allocator warmup, which would otherwise land entirely on whichever
    # engine we happen to measure first.
    print("warming both endpoints...")
    for url in (A_URL, B_URL):
        try:
            requests.post(url, json={"text": "नमस्ते"}, timeout=180)
        except Exception as e:
            print(f"  warmup failed for {url}: {type(e).__name__}")

    a = bench(A_NAME, A_URL, SENTENCES)
    b = bench(B_NAME, B_URL, SENTENCES)

    print(f"\n{'=' * 66}\nVERDICT\n{'=' * 66}")
    if not (a and b):
        print("  Could not benchmark both endpoints.")
        if not a:
            print(f"  {A_NAME} unreachable at {A_URL}")
        if not b:
            print(f"  {B_NAME} unreachable at {B_URL}")
        sys.exit(1)

    af, bf = a[0], b[0]
    am, bm = statistics.median(a), statistics.median(b)
    print(f"  first clause : {A_NAME} {af:6.3f}s    vs    {B_NAME} {bf:6.3f}s")
    print(f"  median       : {A_NAME} {am:6.3f}s    vs    {B_NAME} {bm:6.3f}s")
    print()

    faster, slower = (B_NAME, A_NAME) if bf < af else (A_NAME, B_NAME)
    ratio = max(af, bf) / min(af, bf) if min(af, bf) > 0 else float("nan")
    print(f"  {faster} is {ratio:.2f}x faster than {slower} on the first clause.")
    print()
    print("  Live calls showed FIRST_AUDIO = GEMMA_FIRST_SENTENCE + TTS_FIRST_CHUNK,")
    print("  so the first-clause figure comes straight off every turn. Note this")
    print("  measures TTS alone -- the contention win also shows up as Gemma")
    print("  getting its decode throughput back, which this does not capture.")


if __name__ == "__main__":
    main()
