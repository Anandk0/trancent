import re
import json
import time
import codecs
import queue
import threading
import requests
import uuid
from typing import Generator

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn


# ============================================================
# SERVICE URLS & HTTP SESSION (Connection Pooling)
# ============================================================

ASR_URL = "http://127.0.0.1:8001/transcribe"
GEMMA_URL = "http://127.0.0.1:8000/chat"
GEMMA_STREAM_URL = "http://127.0.0.1:8000/chat/stream"
TTS_URL = "http://127.0.0.1:8003/synthesize"

HOST = "0.0.0.0"
PORT = 8002

# Persistent HTTP session to reuse connections to local services
http_session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=15, pool_maxsize=30)
http_session.mount("http://", adapter)
http_session.mount("https://", adapter)


app = FastAPI(title="MBA Calling Agent API")


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "mba-calling-agent",
        "endpoints": ["/health", "/chat", "/chat/stream"]
    }


# ============================================================
# TEXT SEGMENTER (Sentence Pipelining for TTS)
# ============================================================

def split_into_sentences(text: str) -> list[str]:
    """
    Split Hindi, Hinglish, and English text into natural sentence / clause chunks
    suitable for progressive TTS synthesis.
    Splits on Devanagari danda (।), period (.), question mark (?), exclamation mark (!),
    or newline, while keeping chunks coherent and natural.
    """
    if not text:
        return []
    raw_chunks = re.split(r'([।\.\?\!\n]+)', text)
    sentences = []
    current = ""
    for piece in raw_chunks:
        current += piece
        if re.search(r'[।\.\?\!\n]', piece):
            cleaned = current.strip()
            if cleaned and len(cleaned) > 1:
                sentences.append(cleaned)
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences if sentences else [text.strip()]


# ============================================================
# STRUCTURED GEMMA PROMPT
# ============================================================

def build_structured_prompt(transcript: str) -> str:
    return f"""You are Divya, the AI admissions counselor for Regular MBA at Jain College of Engineering and Research, Udyambag, Belagavi (VTU affiliated, AICTE approved).

Respond directly to the student's message.
Keep your response concise, helpful, and natural for a phone call (1 to 3 sentences maximum).

Return your response as a single valid JSON object with exactly two keys:
- "reply": your natural conversational response matching the student's language style (English, Hindi, or Hinglish).
- "tts_text": the exact same response written in Devanagari script for the Hindi TTS voice.

Rules for tts_text:
- If the reply is in Hindi or Hinglish, write the ENTIRE tts_text in Devanagari script.
- English professional words in Hindi/Hinglish must be written phonetically in Devanagari (e.g., MBA → एमबीए, HR → एचआर, Finance → फाइनेंस, Marketing → मार्केटिंग, Placement → प्लेसमेंट, Admission → एडमिशन, Specialization → स्पेशलाइज़ेशन, Business Analytics → बिज़नेस एनालिटिक्स).
- If the reply is purely in English, tts_text can remain in English.
- Do NOT include unnecessary punctuation around English words.

Specializations available: Marketing, Finance, Human Resource Management, Business Analytics.
Do NOT invent fees, placement percentages, salary figures, recruiter names, or unverified deadlines.

Return ONLY the JSON object. No extra text, no markdown backticks.

Example output:
{{"reply": "Haan, MBA ka duration 2 saal ka hai. Aap kaunsi specialization mein interested hain?", "tts_text": "हाँ, एमबीए का ड्यूरेशन 2 साल का है। आप कौनसी स्पेशलाइज़ेशन में इंटरेस्टेड हैं?"}}

Student message:
{transcript}"""


# ============================================================
# PARSE GEMMA STRUCTURED RESPONSE
# ============================================================

def parse_gemma_json(raw: str, fallback_reply: str = "") -> tuple:
    """
    Parse the JSON object from Gemma's raw output.
    Returns (reply, tts_text).
    """
    if not raw or not raw.strip():
        return fallback_reply, fallback_reply

    text = raw.strip()

    # Strip markdown code fences if present e.g. ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # 1. Try standard JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            reply = str(data.get("reply", "") or "").strip()
            tts_text = str(data.get("tts_text", "") or "").strip()

            if not reply:
                reply = fallback_reply
            if not tts_text:
                tts_text = reply

            return reply, tts_text
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 2. Try extracting JSON object substring {...}
    brace_match = re.search(r"(\{[\s\S]*\})", text)
    if brace_match:
        try:
            data = json.loads(brace_match.group(1))
            if isinstance(data, dict):
                reply = str(data.get("reply", "") or "").strip()
                tts_text = str(data.get("tts_text", "") or "").strip()

                if not reply:
                    reply = fallback_reply
                if not tts_text:
                    tts_text = reply

                return reply, tts_text
        except Exception:
            pass

    # 3. Try regex extraction for "reply" and "tts_text"
    reply_match = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    tts_match = re.search(r'"tts_text"\s*:\s*"((?:[^"\\]|\\.)*)"', text)

    reply = ""
    tts_text = ""

    if reply_match:
        reply = reply_match.group(1).replace('\\"', '"').replace('\\n', ' ').strip()
    if tts_match:
        tts_text = tts_match.group(1).replace('\\"', '"').replace('\\n', ' ').strip()

    if reply or tts_text:
        if not reply:
            reply = fallback_reply
        if not tts_text:
            tts_text = reply
        print("[WARN] JSON parse recovered fields via regex.")
        return reply, tts_text

    # Complete fallback
    reply = text or fallback_reply
    print("[WARN] JSON parse failed; using raw Gemma output.")
    return reply, reply


# ============================================================
# STREAMING-FRIENDLY GEMMA PROMPT (/chat/stream only)
#
# The structured JSON contract used by /chat cannot be parsed until the
# closing brace arrives, which forces the caller to wait for the entire
# Gemma response. For /chat/stream we instead ask Gemma for plain
# conversational text only (already Devanagari-normalized per the same
# rules), so partial output is usable the moment it arrives and no second
# LLM call is needed to normalize it for TTS.
# ============================================================

def build_streaming_prompt(transcript: str) -> str:
    return f"""You are Divya, the AI admissions counselor for Regular MBA at Jain College of Engineering and Research, Udyambag, Belagavi (VTU affiliated, AICTE approved).

Respond directly to the student's message.
Keep your response concise, helpful, and natural for a phone call (1 to 3 sentences maximum).

Return ONLY the spoken reply text. Do NOT use JSON, markdown, or any labels.

If the reply is in Hindi or Hinglish, write the ENTIRE reply in Devanagari script.
English professional words in Hindi/Hinglish must be written phonetically in Devanagari (e.g., MBA → एमबीए, HR → एचआर, Finance → फाइनेंस, Marketing → मार्केटिंग, Placement → प्लेसमेंट, Admission → एडमिशन, Specialization → स्पेशलाइज़ेशन, Business Analytics → बिज़नेस एनालिटिक्स).
If the reply is purely in English, keep it in English.
Do NOT include unnecessary punctuation around English words.

Specializations available: Marketing, Finance, Human Resource Management, Business Analytics.
Do NOT invent fees, placement percentages, salary figures, recruiter names, or unverified deadlines.

Student message:
{transcript}"""


_SENTENCE_BOUNDARY_RE = re.compile(r'[।\.\?\!\n]+')


def extract_complete_sentences(buffer_text: str) -> tuple:
    """
    Given the accumulated (not-yet-flushed) streamed text, split off every
    complete sentence/clause ending in a boundary character, and return the
    remaining unterminated tail so it can keep accumulating.
    """
    matches = list(_SENTENCE_BOUNDARY_RE.finditer(buffer_text))
    if not matches:
        return [], buffer_text
    last_end = matches[-1].end()
    sentences = split_into_sentences(buffer_text[:last_end])
    return sentences, buffer_text[last_end:]


# For the FIRST audio chunk only, flush at the earliest natural break — a comma
# or clause boundary — instead of waiting for a full sentence. Gemma then only
# has to generate a handful of words, and TTS only has to synthesize a short
# clause, so the caller hears the first audio in ~1-2s instead of waiting for a
# long opening sentence. Later chunks use full sentence boundaries for natural
# prosody.
_FIRST_CHUNK_BOUNDARY_RE = re.compile(r'[,;:।\.\?\!\n]')
_FIRST_CHUNK_MIN_LEN = 6
_FIRST_CHUNK_MAX_LEN = 45


def extract_first_chunk(buffer_text: str, min_len: int = _FIRST_CHUNK_MIN_LEN,
                        max_len: int = _FIRST_CHUNK_MAX_LEN) -> tuple:
    """
    Return (chunk, remaining) as soon as the first audio chunk can be flushed;
    otherwise (None, buffer) to keep accumulating.

    Flush when either:
      - an early clause boundary (comma/danda/etc.) appears at/after min_len, or
      - the buffer passes max_len with no boundary yet — then cut at the last
        space so a long comma-less opening clause still yields fast first audio
        without splitting a word.
    """
    for m in _FIRST_CHUNK_BOUNDARY_RE.finditer(buffer_text):
        if m.end() >= min_len:
            return buffer_text[:m.end()].strip(), buffer_text[m.end():]

    if len(buffer_text) >= max_len:
        cut = buffer_text.rfind(" ")
        if cut >= min_len:
            return buffer_text[:cut].strip(), buffer_text[cut:]

    return None, buffer_text


def stream_gemma_text(session_id: str, prompt: str, timings: dict):
    """
    Opens a streaming HTTP request to the Gemma service's /chat/stream route
    and yields plain text chunks as tokens arrive on the socket, instead of
    buffering the whole response before returning (as `response.json()` on the
    batch /chat route would force).

    /chat/stream on gemma_server.py drives model.generate() through a
    TextIteratorStreamer on a background thread, so tokens are emitted as the
    model produces them. This function still keeps a safety detector: if the
    backend ever responds with one fully-buffered JSON blob (the batch /chat
    contract: {"reply": "..."}) instead of genuine token text, it records that
    in `timings['gemma_genuinely_streaming']` plus `timings['gemma_chunk_count']`
    and recovers the reply, so the caller can honestly report what happened.
    """
    t_start = timings["t_request_start"]

    response = http_session.post(
        GEMMA_STREAM_URL,
        json={"session_id": session_id, "text": prompt, "stream": True},
        stream=True,
        timeout=120,
    )
    response.raise_for_status()

    decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
    raw_accum = ""
    chunk_count = 0
    is_json_mode = None  # decided once the first non-whitespace char is seen

    for raw_bytes in response.iter_content(chunk_size=64):
        if not raw_bytes:
            continue

        piece = decoder.decode(raw_bytes)
        if not piece:
            continue

        now = time.perf_counter() - t_start
        if "gemma_first_token_s" not in timings:
            timings["gemma_first_token_s"] = round(now, 3)

        raw_accum += piece
        chunk_count += 1

        if is_json_mode is None:
            stripped = raw_accum.lstrip()
            if stripped:
                is_json_mode = stripped.startswith("{") or stripped.startswith("```")

        if not is_json_mode:
            if "gemma_first_text_chunk_s" not in timings:
                timings["gemma_first_text_chunk_s"] = round(now, 3)
            yield piece

    tail = decoder.decode(b"", final=True)
    if tail:
        raw_accum += tail
        if not is_json_mode:
            yield tail

    timings["gemma_total_s"] = round(time.perf_counter() - t_start, 3)
    timings["gemma_chunk_count"] = chunk_count
    # Genuinely streaming only if plain text arrived across multiple reads;
    # a single chunk (or JSON that must be parsed whole) means the backend
    # buffered the full generation before responding.
    timings["gemma_genuinely_streaming"] = bool(not is_json_mode and chunk_count > 1)

    if is_json_mode:
        reply, _ = parse_gemma_json(raw_accum, fallback_reply=raw_accum.strip())
        if "gemma_first_text_chunk_s" not in timings:
            timings["gemma_first_text_chunk_s"] = timings["gemma_total_s"]
        print("[WARN] Gemma backend returned buffered JSON on /chat/stream; no genuine token streaming observed.")
        yield reply


# ============================================================
# MAIN BATCH CHAT ENDPOINT (Existing Fallback Pipeline)
# ============================================================

@app.post("/chat")
async def chat(
    file: UploadFile = File(...),
    session_id: str = Form(None),
    language: str = Form("hi")
):
    if not session_id:
        session_id = str(uuid.uuid4())

    t_request_start = time.perf_counter()

    print("\n" + "=" * 60)
    print("NEW CHAT REQUEST (BATCH)")
    print("Session:", session_id)
    print("=" * 60)

    # 1. AUDIO → ASR
    print("\n[1/3] Sending audio to ASR...")
    t_read_start = time.perf_counter()
    audio_data = await file.read()
    t_read_elapsed = time.perf_counter() - t_read_start

    t_asr_start = time.perf_counter()
    try:
        asr_response = http_session.post(
            ASR_URL,
            files={
                "file": (
                    file.filename or "audio.wav",
                    audio_data,
                    file.content_type or "audio/wav"
                )
            },
            data={
                "language": language,
                "decoding": "ctc"
            },
            timeout=120
        )
        asr_response.raise_for_status()
        asr_result = asr_response.json()
    except Exception as e:
        print("ASR ERROR:", e)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "stage": "asr",
                "error": str(e)
            }
        )

    t_asr_elapsed = time.perf_counter() - t_asr_start
    transcript = asr_result.get("text", "").strip()
    print(f"[TIMING] ASR: {t_asr_elapsed:.3f}s -> Transcript: '{transcript}'")

    if not transcript:
        return {
            "status": "success",
            "session_id": session_id,
            "language": language,
            "transcript": "",
            "reply": "",
            "tts_text": "",
            "audio_file": None,
            "audio_b64": None
        }

    # 2. TRANSCRIPT → GEMMA
    print("\n[2/3] Sending transcript to Gemma...")
    structured_prompt = build_structured_prompt(transcript)

    t_gemma_start = time.perf_counter()
    try:
        gemma_response = http_session.post(
            GEMMA_URL,
            json={
                "session_id": session_id,
                "text": structured_prompt
            },
            timeout=120
        )
        gemma_response.raise_for_status()
        gemma_result = gemma_response.json()
    except Exception as e:
        print("GEMMA ERROR:", e)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "stage": "gemma",
                "transcript": transcript,
                "error": str(e)
            }
        )

    t_gemma_elapsed = time.perf_counter() - t_gemma_start
    raw_gemma = gemma_result.get("reply", "").strip()
    reply, tts_text = parse_gemma_json(raw_gemma, fallback_reply=transcript)

    print(f"[TIMING] GEMMA: {t_gemma_elapsed:.3f}s")
    print(f"  Reply:    '{reply}'")
    print(f"  TTS text: '{tts_text}'")

    # 3. DEVANAGARI → PARLER TTS
    print("\n[3/3] Sending text to Parler TTS...")
    t_tts_start = time.perf_counter()
    try:
        tts_response = http_session.post(
            TTS_URL,
            json={
                "text": tts_text
            },
            timeout=120
        )
        tts_response.raise_for_status()
        tts_result = tts_response.json()
    except Exception as e:
        print("TTS ERROR:", e)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "stage": "tts",
                "session_id": session_id,
                "transcript": transcript,
                "reply": reply,
                "tts_text": tts_text,
                "error": str(e)
            }
        )

    t_tts_elapsed = time.perf_counter() - t_tts_start
    audio_file = tts_result.get("audio_file")
    audio_b64 = tts_result.get("audio_b64")
    sample_rate = tts_result.get("sample_rate", 22050)

    t_total = time.perf_counter() - t_request_start

    print("\n" + "=" * 60)
    print("REQUEST COMPLETE (BATCH)")
    print(f"Session:     {session_id}")
    print(f"Audio bytes: {len(audio_data)}")
    print(f"ASR:         {t_asr_elapsed:.3f} s")
    print(f"Gemma:       {t_gemma_elapsed:.3f} s")
    print(f"TTS:         {t_tts_elapsed:.3f} s")
    print(f"I/O:         {t_read_elapsed:.3f} s")
    print(f"TOTAL:       {t_total:.3f} s")
    print("=" * 60 + "\n")

    return {
        "status": "success",
        "session_id": session_id,
        "language": language,
        "transcript": transcript,
        "reply": reply,
        "tts_text": tts_text,
        "audio_file": audio_file,
        "audio_b64": audio_b64,
        "sample_rate": sample_rate,
        "timings": {
            "asr_s": round(t_asr_elapsed, 3),
            "gemma_s": round(t_gemma_elapsed, 3),
            "tts_s": round(t_tts_elapsed, 3),
            "total_s": round(t_total, 3)
        }
    }


# ============================================================
# STREAMING SSE CHAT ENDPOINT (Sentence-Pipelined SSE)
# ============================================================

@app.post("/chat/stream")
async def chat_stream(
    file: UploadFile = File(None),
    transcript: str = Form(None),
    session_id: str = Form(None),
    language: str = Form("hi")
):
    if not session_id:
        session_id = str(uuid.uuid4())

    # Read the upload once, up front, so the SSE body can be a synchronous
    # generator (Starlette runs sync generators in a worker thread, which lets
    # us safely block on a Queue while a producer thread reads Gemma tokens).
    transcript_arg = transcript
    audio_bytes = await file.read() if file is not None else None
    audio_filename = (file.filename or "audio.wav") if file is not None else "audio.wav"
    audio_content_type = (file.content_type or "audio/wav") if file is not None else "audio/wav"

    # Sentinel marking end-of-stream on the sentence queue.
    _QUEUE_DONE = object()

    def event_generator() -> Generator[str, None, None]:
        t_request_start = time.perf_counter()
        active_transcript = (transcript_arg or "").strip()
        t_asr = 0.0

        print("\n" + "=" * 60)
        print("NEW STREAMING CHAT REQUEST")
        print("Session:", session_id)
        print("=" * 60)

        # Step 1: ASR if audio provided and no transcript override
        if not active_transcript and audio_bytes is not None:
            t_asr_start = time.perf_counter()
            try:
                asr_response = http_session.post(
                    ASR_URL,
                    files={"file": (audio_filename, audio_bytes, audio_content_type)},
                    data={"language": language, "decoding": "ctc"},
                    timeout=120
                )
                asr_response.raise_for_status()
                active_transcript = asr_response.json().get("text", "").strip()
            except Exception as e:
                print("ASR STREAM ERROR:", e)
                yield f"data: {json.dumps({'type': 'error', 'stage': 'asr', 'error': str(e)})}\n\n"
                return

            t_asr = time.perf_counter() - t_asr_start
            print(f"[TIMING] ASR: {t_asr:.3f}s -> '{active_transcript}'")
            yield f"data: {json.dumps({'type': 'asr_final', 'transcript': active_transcript, 'asr_time_s': round(t_asr, 3)})}\n\n"

        if not active_transcript:
            yield f"data: {json.dumps({'type': 'done', 'reply': '', 'tts_text': '', 'audio_chunks': 0})}\n\n"
            return

        # Step 2: Producer/consumer streaming pipeline.
        #
        #   Gemma producer thread          main (consumer) generator
        #   --------------------           -------------------------
        #   read /chat/stream tokens        block on sentence_queue.get()
        #   accumulate sentence buffer      TTS-synthesize each sentence in
        #   on sentence boundary: put()  →  FIFO order (single worker, so audio
        #   ...continue reading Gemma       chunks stay correctly ordered)
        #                                   yield audio_chunk SSE
        #
        # Because the producer keeps reading Gemma while the consumer is busy
        # synthesizing, TTS(sentence 1) overlaps Gemma generating sentence 2.
        yield f"data: {json.dumps({'type': 'gemma_start'})}\n\n"
        streaming_prompt = build_streaming_prompt(active_transcript)

        timings = {"t_request_start": t_request_start}
        sentence_queue: "queue.Queue" = queue.Queue()
        producer_state = {"reply_parts": [], "error": None}

        def gemma_producer():
            buffer = ""
            first_chunk_done = False
            try:
                for text_piece in stream_gemma_text(session_id, streaming_prompt, timings):
                    producer_state["reply_parts"].append(text_piece)
                    buffer += text_piece

                    # First audio chunk: flush at the earliest clause boundary so
                    # the caller hears something in ~1-2s.
                    if not first_chunk_done:
                        chunk, buffer = extract_first_chunk(buffer)
                        if chunk:
                            first_chunk_done = True
                            timings["gemma_first_sentence_s"] = round(time.perf_counter() - t_request_start, 3)
                            print(f"[TIMING] GEMMA_FIRST_TOKEN: {timings.get('gemma_first_token_s', 0):.3f}s")
                            print(f"[TIMING] GEMMA_FIRST_TEXT_CHUNK: {timings.get('gemma_first_text_chunk_s', 0):.3f}s")
                            print(f"[TIMING] GEMMA_FIRST_SENTENCE: {timings['gemma_first_sentence_s']:.3f}s")
                            sentence_queue.put(chunk)
                        continue

                    # Subsequent chunks: full sentence boundaries for natural prosody.
                    sentences, buffer = extract_complete_sentences(buffer)
                    for sentence in sentences:
                        sentence_queue.put(sentence)

                # Flush any trailing text that never hit a boundary.
                tail = buffer.strip()
                if tail:
                    if "gemma_first_sentence_s" not in timings:
                        timings["gemma_first_sentence_s"] = round(time.perf_counter() - t_request_start, 3)
                    sentence_queue.put(tail)
            except Exception as e:
                producer_state["error"] = str(e)
                print("GEMMA STREAM ERROR:", e)
            finally:
                sentence_queue.put(_QUEUE_DONE)

        producer = threading.Thread(target=gemma_producer, name="gemma-producer", daemon=True)
        producer.start()

        # Consumer: drain sentences in order, one TTS call at a time.
        chunk_index = 0
        first_audio_ready = False
        t_tts_first_chunk = 0.0
        t_first_audio_ready = 0.0

        while True:
            sentence = sentence_queue.get()
            if sentence is _QUEUE_DONE:
                break

            t_tts_chunk_start = time.perf_counter()
            try:
                tts_response = http_session.post(TTS_URL, json={"text": sentence}, timeout=120)
                tts_response.raise_for_status()
                tts_result = tts_response.json()
                audio_b64 = tts_result.get("audio_b64")
                sample_rate = tts_result.get("sample_rate", 22050)
                t_tts_chunk_elapsed = time.perf_counter() - t_tts_chunk_start

                if not first_audio_ready:
                    first_audio_ready = True
                    t_tts_first_chunk = t_tts_chunk_elapsed
                    t_first_audio_ready = time.perf_counter() - t_request_start
                    print(f"[TIMING] TTS_FIRST_CHUNK: {t_tts_first_chunk:.3f}s")
                    print(f"[TIMING] FIRST_AUDIO_READY: {t_first_audio_ready:.3f}s")

                chunk_payload = {
                    "type": "audio_chunk",
                    "chunk_index": chunk_index,
                    "total_chunks": None,
                    "sentence": sentence,
                    "audio_b64": audio_b64,
                    "sample_rate": sample_rate,
                    "tts_time_s": round(t_tts_chunk_elapsed, 3),
                    "first_audio_latency_s": round(t_first_audio_ready, 3)
                }
                yield f"data: {json.dumps(chunk_payload)}\n\n"
                chunk_index += 1
            except Exception as e:
                print(f"TTS CHUNK {chunk_index} ERROR:", e)
                yield f"data: {json.dumps({'type': 'tts_chunk_error', 'chunk_index': chunk_index, 'error': str(e)})}\n\n"
                chunk_index += 1

        producer.join()

        # If Gemma streaming failed and produced no audio, surface the error.
        if producer_state["error"] and chunk_index == 0:
            yield f"data: {json.dumps({'type': 'error', 'stage': 'gemma', 'error': producer_state['error']})}\n\n"
            return

        reply = "".join(producer_state["reply_parts"]).strip()
        t_gemma_elapsed = timings.get("gemma_total_s", round(time.perf_counter() - t_request_start, 3))
        print(f"[TIMING] GEMMA_TOTAL: {t_gemma_elapsed:.3f}s")
        print(f"[TIMING] GEMMA_GENUINELY_STREAMING: {timings.get('gemma_genuinely_streaming', False)} "
              f"({timings.get('gemma_chunk_count', 0)} chunk(s) received)")
        yield f"data: {json.dumps({'type': 'gemma_final', 'reply': reply, 'tts_text': reply, 'gemma_time_s': t_gemma_elapsed})}\n\n"

        t_agent_total = time.perf_counter() - t_request_start
        print(f"[TIMING] AGENT_TOTAL: {t_agent_total:.3f}s")
        print("=" * 60 + "\n")

        done_payload = {
            "type": "done",
            "session_id": session_id,
            "transcript": active_transcript,
            "reply": reply,
            "tts_text": reply,
            "total_chunks": chunk_index,
            "timings": {
                "asr_s": round(t_asr, 3),
                "gemma_first_token_s": timings.get("gemma_first_token_s"),
                "gemma_first_text_chunk_s": timings.get("gemma_first_text_chunk_s"),
                "gemma_first_sentence_s": timings.get("gemma_first_sentence_s"),
                "gemma_s": t_gemma_elapsed,
                "gemma_genuinely_streaming": timings.get("gemma_genuinely_streaming", False),
                "tts_first_chunk_s": round(t_tts_first_chunk, 3),
                "first_audio_ready_s": round(t_first_audio_ready, 3),
                "agent_total_s": round(t_agent_total, 3)
            }
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT
    )
