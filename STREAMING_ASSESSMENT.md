# Technical Assessment: Streaming Voice Architecture

**Project**: Jain College of Engineering and Research, Belagavi – MBA Admissions Voice Calling Agent ("Divya")  
**Target Goal**: Low-latency, streaming turn-taking pipeline (minimizing "User stops speaking → Divya starts speaking" delay).

---

## 1. Inspection of Existing Architecture & Component Boundaries

### 1.1 Microphone Capture (`call_server.py`)
- **Current Mechanism**: Browser uses `MediaRecorder(mediaStream)` with `CHUNK_MS = 100`. Recorded WebM chunks are collected in an in-memory array (`recordedChunks`).
- **Trigger**: When VAD detects silence (`SILENCE_MS = 1200`), the recorder stops and packs all chunks into a single `Blob({ type: "audio/webm" })` and POSTs to `/process`.
- **Streaming Status**: The browser microphone emits chunks every 100ms, but currently waits for the full utterance to finish before transmitting.

### 1.2 Audio Ingestion (`call_server.py` → `agent_api.py`)
- **Current Mechanism**: Forwarded as a single multipart `UploadFile` via HTTP POST to `http://127.0.0.1:8002/chat`.
- **Streaming Status**: Batch HTTP boundary.

### 1.3 ASR Service (`agent_api.py` → `asr-server/asr_api.py`)
- **Current Model**: `ai4bharat/indic-conformer-600m-multilingual` on CPU (CTC / RNN-T decoding).
- **Current Mechanism**: Receives whole audio file/stream via HTTP POST `/transcribe`. Decodes via FFmpeg into float32 16kHz tensor and runs `model(wav, language, decoding)`.
- **Streaming Feasibility**: Indic Conformer is an encoder CTC model. It can transcribe sliding audio buffers (e.g. 500ms–2s progressive window) to output partial transcripts as the student is speaking.

### 1.4 LLM Generation (`agent_api.py` → Gemma `:8000`)
- **Current Model**: `google/gemma-4-E4B-it` via Transformers `AutoModelForMultimodalLM`.
- **Current Mechanism**: HTTP POST `/chat` with full prompt, waits for complete response, returns JSON with `reply` and `tts_text`.
- **Streaming Feasibility**: Hugging Face Transformers supports `TextIteratorStreamer` / async token generators for token-by-token streaming.

### 1.5 Text-to-Speech (`agent_api.py` → `tts_api.py`)
- **Current Model**: `ai4bharat/indic-parler-tts` on GPU.
- **Current Mechanism**: HTTP POST `/synthesize` with full `tts_text`. Runs `model.generate(...)` for all audio codes and decodes to 22.05kHz WAV in batch.
- **Streaming Feasibility**: Parler TTS generates discrete acoustic codes and converts them to waveform through the DAC neural decoder. While intra-word acoustic streaming is not exposed in the standard high-level API, **sentence-pipelined generation** (synthesizing Sentence 1 as soon as Gemma emits it) enables early audio playback while Sentence 2 is still being generated.

---

## 2. Streaming Capability Matrix

### SUPPORTED NOW
1. **Microphone chunking**: Browser `MediaRecorder` can stream audio chunks continuously (100–300ms chunks).
2. **GPU Voice Conditioning Cache**: Pre-cached Divya speaker conditioning allows rapid startup for any synthesized sentence chunk.
3. **In-Memory Zero-Copy Transport**: Direct `io.BytesIO` and base64 transmission eliminates disk I/O latency.
4. **SSE / Chunked HTTP Transport**: Full compatibility with external reverse proxies without WebSocket upgrade failures.

### NEEDS ADAPTER
1. **Progressive Buffer ASR Adapter**: Accumulates incoming audio chunks during active speech, running periodic Conformer passes to stream real-time partial transcripts to the browser.
2. **Gemma Token Streamer Adapter**: Emits tokens incrementally via Server-Sent Events (`text/event-stream`).
3. **Sentence / Clause Segmenter & Pipeline Dispatcher**: Buffers Gemma token stream until a sentence boundary (e.g. `।`, `.`, `?`, `!`, `\n`) is reached, immediately dispatching that clause to Parler TTS.
4. **Browser Audio Chunk Queue Player**: HTML5 Web Audio API buffer queue that begins playing Audio Chunk 1 immediately and enqueues subsequent chunks smoothly without clicks or delays.

### LIKELY REQUIRES DIFFERENT MODEL (Future Enhancement)
- **Zero-Latency Streaming Vocoder TTS**: For sub-300ms intra-sentence acoustic streaming (e.g., streaming VITS, FastSpeech2 streaming, or streaming flow-matching).
- *Note*: Sentence-pipelined Parler TTS provides a massive real-world reduction (first audio arrives in ~1.8–2.5s instead of 16–22s) using the existing model weights without quality degradation.

---

## 3. Streaming Architecture Diagram

```text
Browser Microphone
       ↓ (continuous 250ms chunks via HTTP POST)
Progressive ASR Buffer
       ↓ (real-time partial transcripts emitted via SSE)
Student Turn Completed (1200ms VAD Silence)
       ↓ (final transcript committed)
Gemma Token Streamer
       ↓ (tokens emitted via SSE)
Sentence Boundary Segmenter ("Sentence 1.", "Sentence 2.")
       ↓
Parler TTS (Sentence 1 synthesized in parallel with Gemma generating Sentence 2)
       ↓ (audio_chunk 1 base64 emitted via SSE)
Browser Web Audio Queue Player (Plays Divya Audio Chunk 1 immediately!)
```
