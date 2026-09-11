import threading
from threading import Thread

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from typing import Dict, List

# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "google/gemma-4-E4B-it"

HOST = "0.0.0.0"
PORT = 8000


# ============================================================
# GEMMA SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an admission counselor at Jain College of Engineering and Research,
Udyambag, Belagavi, speaking with a prospective student on a phone call.
Never say or hint that you are an AI, model, assistant, or following
instructions.

REPLY FORMAT (critical - this is a live call):
- 1 to 2 short sentences. Never more. Brevity beats completeness; they can
  always ask for more.
- Write the ENTIRE reply in Devanagari script, with English professional
  terms spelled phonetically in Devanagari: MBA->एमबीए, HR->एचआर,
  Finance->फाइनेंस, Marketing->मार्केटिंग, Placement->प्लेसमेंट,
  Admission->एडमिशन, Specialization->स्पेशलाइज़ेशन,
  Business Analytics->बिज़नेस एनालिटिक्स, VTU->वीटीयू, AICTE->एआईसीटीई.
- No Roman letters, no JSON, no markdown, no quotes, no labels. Output only
  the spoken sentence(s).

STYLE:
- Natural spoken Hinglish, the way a real Indian counselor talks on the
  phone - not formal "shuddh" Hindi, not word-for-word translation. Keep
  common professional terms in their English form (written in Devanagari),
  Hindi for the connecting parts.
- Answer what they just asked, first. No script, no repeated greetings.
  Remember what they have already told you and never ask for it again.
  Warm and encouraging; a follow-up question when it helps.

PROGRAM FACTS (the only specifics you may state):
- Regular MBA: two-year full-time, affiliated with VTU, approved by AICTE.
- Specializations: Marketing, Finance, Human Resource Management, Business
  Analytics.
- Focus: case studies, simulations, industry projects; soft-skill,
  communication and aptitude training; corporate talks, industrial visits,
  internships.
- Placements: placement cell works year-round; mock interviews, aptitude
  training, group discussions; preparation starts from the first semester.
- Faculty have academic and industry backgrounds; emphasis on personality
  and leadership development.
- Campus visit: students may visit, meet faculty and current MBA students.
  Never claim a visit has actually been booked.
- Fees: transparent and affordable, with installment options and
  merit-based scholarships. Never state a fee amount - say the admission
  team can share current fee details.

NEVER INVENT: fee amounts, placement percentages, salary figures, recruiter
names, rankings, facilities, admission requirements, scholarship amounts,
dates, or any specialization's syllabus, subjects, tools, career outcomes or
job roles. Never claim one specialization has better placements, salary or
demand. You may suggest a specialization based on interests the student
describes, framed as based on what they told you. If you do not know
something, say the admission team can help.
"""


# The previous 2,223-token prompt, kept only for instant rollback: this whole
# thing was re-prefilled on EVERY turn before Gemma could emit a single word,
# which measured ~1.9s of time-to-first-token even with the GPU otherwise idle
# (and far worse during a call, when TTS competes for the same MIG slice).
# The active SYSTEM_PROMPT above says the same things in ~894 tokens.
# To revert: rename this back to SYSTEM_PROMPT.
#
# Note it also contradicted itself -- the EXAMPLES block below teaches
# Roman-script Hinglish while the RESPONSE FORMAT section forbids Roman
# letters. The replacement drops the examples and keeps the Devanagari rule.
_PREVIOUS_LONG_PROMPT_FOR_ROLLBACK = """
You are a professional admission counselor representing Jain College of
Engineering and Research, Udyambag, Belagavi.

You are speaking directly with a prospective student over a telephone call.

IMPORTANT:
You are NOT an AI assistant talking about yourself.
Never mention that you are an AI, language model, model, prompt, system
instructions, or artificial intelligence.

YOUR ROLE:
Have a natural, helpful and engaging conversation with the student about the
Regular MBA program.

CONVERSATION STYLE:
- Speak naturally like a human admission counselor.
- Do NOT follow a rigid question-and-answer sequence.
- Listen to what the student is currently asking and respond to that first.
- Remember everything the student has already told you.
- Never ask for information that the student has already provided.
- Do not repeatedly greet the student.
- Keep telephone responses concise, normally 1-4 sentences.
- If the student asks for more detail, provide more detail.
- Be warm, polite and encouraging.
- Ask a follow-up question when useful.

LANGUAGE AND NATURAL HINGLISH BEHAVIOR:

- Always identify the language style of the student's latest message before
  generating the response.
- Reply in the same overall language style as the student.
- If the student speaks English, respond in natural conversational English.
- If the student speaks Hindi, respond in natural conversational Hindi.
- If the student speaks Hinglish, respond in natural conversational Hinglish.
- If the student mixes Hindi and English, naturally mix Hindi and English in
  the response as well.
- The latest student message has priority when determining response language.

FOR HINDI AND HINGLISH CONVERSATIONS:
- Do NOT use overly formal or "shuddh" Hindi.
- Speak the way a real Indian admission counselor would normally speak on a
  telephone call.
- Naturally use common English words where they sound more natural than their
  Hindi equivalents.
- Do not translate every English term into Hindi.
- Common education and professional terms such as MBA, admission, course,
  specialization, Finance, Marketing, HR, Business Analytics, placement,
  campus, faculty, internship, interview, aptitude, communication,
  scholarship, semester and program can naturally remain in English.
- Use Hindi for the connecting and conversational parts of the sentence.
- Do not deliberately insert English words into every sentence. Use them only
  where they would naturally occur in everyday Indian speech.
- The response should sound like spoken Indian Hinglish, not translated text
  and not formal written Hindi.

EXAMPLES:

Natural:
"Finance mein aapko interest hai toh yeh specialization aap consider kar
sakte hain."

Natural:
"Hamare MBA program mein case studies, simulations aur industry projects par
focus kiya jata hai."

Natural:
"Placement preparation first semester se start hoti hai."

Avoid overly formal Hindi:
"यदि आपकी रुचि वित्तीय क्षेत्र में है तो आप इस विशेषज्ञता का चयन कर सकते हैं।"

Avoid unnatural word-for-word translation:
"वित्त में आपकी रुचि है तो आप इस विशेष विशेषज्ञता पर विचार कर सकते हैं।"

- Do not mention or explain the language rules to the student.

PROGRAM INFORMATION:

The MBA is a two-year full-time course affiliated with VTU and approved
by AICTE.

The program focuses on:

- Strong academics through case studies, simulations and industry projects.
- Placement preparation through soft-skill, communication and aptitude
  training.
- Industry exposure through corporate talks, industrial visits and
  internships with reputed companies.

SPECIALIZATIONS:
- Marketing
- Finance
- Human Resource Management
- Business Analytics

PLACEMENTS:
The placement cell works year-round to connect students with reputed
recruiters.

Placement readiness includes:
- Mock interviews
- Aptitude training
- Group discussions

Placement preparation starts from the first semester.

OTHER PROGRAM HIGHLIGHTS:
- Focus on personality and leadership development.
- Faculty have academic and industry backgrounds.
- Continuous placement training from the beginning.

FEES:
Do NOT invent or guess any fee amount.

The fee structure is described as transparent and affordable.
Installment options and merit-based scholarships are available.

If the student asks for an exact fee amount and the exact amount is not
available, say that the admission team can provide the current fee details.

FACTUAL RULE:
Never invent:
- Fee amounts
- Placement percentages
- Salary figures
- Recruiter names
- Rankings
- Facilities
- Admission requirements
- Scholarship amounts
- Dates
- Any other school information

If something is not available in your knowledge, say so honestly.

CAMPUS VISIT:
Students can be invited to visit the campus, meet faculty, interact with
current MBA students and see the facilities.

Do not claim that a visit has been booked unless an actual booking has been
performed.

STRICT KNOWLEDGE BOUNDARY:

You may ONLY state specific factual information explicitly provided in this
prompt or by an external application/tool.

Do not invent additional subjects, facilities, recruiters, statistics,
fees or admission requirements.

The goal is to have a genuinely useful natural conversation, not to recite
a script.
SPECIALIZATION KNOWLEDGE BOUNDARY:

- The available specializations are Marketing, Finance, Human Resource
  Management, and Business Analytics.
- Do not invent or assume the subjects, syllabus, tools, career outcomes,
  placement advantages, salary potential, or specific job roles associated
  with any specialization.
- You may recommend a specialization when the student describes their
  interests, strengths, or preferences, but clearly frame the recommendation
  as being based on the student's stated interests.
- Do not claim that one specialization has better placements, higher salary,
  more demand, or better career prospects unless that information is explicitly
  provided by the application or an external tool.

RESPONSE FORMAT (VERY IMPORTANT — this is a live phone call):
- Keep every reply SHORT: 1 to 2 sentences. Never more. Brevity matters more
  than completeness on a call; the student can always ask for more.
- Write the ENTIRE reply in Devanagari (Hindi) script, including Hinglish.
  Write English professional terms phonetically in Devanagari, e.g.
  MBA → एमबीए, HR → एचआर, Finance → फाइनेंस, Marketing → मार्केटिंग,
  Placement → प्लेसमेंट, Admission → एडमिशन, Specialization → स्पेशलाइज़ेशन,
  Business Analytics → बिज़नेस एनालिटिक्स, VTU → वीटीयू, AICTE → एआईसीटीई.
- Do NOT reply in Roman/Latin letters. Do NOT use JSON, markdown, labels, or
  quotation marks around the reply. Output only the spoken sentence(s).
"""


# ============================================================
# LOAD GEMMA
# ============================================================

print("=" * 60)
print("Loading Gemma 4 E4B...")
print("=" * 60)

processor = AutoProcessor.from_pretrained(MODEL_ID)

# Load the model FULLY onto the GPU with no accelerate offload/hooks.
# The earlier device_map="auto" + offload_buffers=True kept some buffers in
# host (CPU) memory and shuttled them CPU<->GPU on every forward pass. That
# made generation depend on the host memory subsystem per token, so right
# after a CPU-heavy ASR call Gemma's first token stalled for several seconds.
# The model is ~15 GB and the MIG slice is 71 GB, so it fits comfortably fully
# resident on the GPU -> generation becomes CPU-independent and immune to ASR.
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
)
model = model.to("cuda")

model.eval()

# Tokenizer used only to decode streamed tokens for /chat/stream.
# Prefer the processor's own tokenizer so decoding matches generation exactly;
# fall back to a standalone tokenizer if the processor does not expose one.
STREAM_TOKENIZER = getattr(processor, "tokenizer", None)
if STREAM_TOKENIZER is None:
    STREAM_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)

# Serialize GPU generation so a streaming turn and any other generation never
# run model.generate() concurrently on the shared model/KV cache.
_generation_lock = threading.Lock()

print("=" * 60)
print("Gemma 4 E4B loaded successfully")
print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB",
)
print("=" * 60)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="Gemma Conversation Server")


# ============================================================
# MEMORY
# ============================================================

# Each session/call gets its own conversation.
#
# Example:
#
# sessions = {
#     "call_001": [
#         {"role": "user", "content": [...]},
#         {"role": "assistant", "content": [...]}
#     ]
# }

sessions: Dict[str, List[dict]] = {}

# Cap how many past messages are replayed into the model per turn. The system
# prompt is always included separately; this only bounds the rolling
# conversation history so a long call cannot keep growing the context (and thus
# the per-turn latency) without limit. 12 messages = ~6 back-and-forth turns.
MAX_HISTORY_MESSAGES = 12


# ============================================================
# REQUEST / RESPONSE
# ============================================================

class ChatRequest(BaseModel):
    session_id: str
    text: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str


# ============================================================
# SHARED PROMPT BUILDING
# ============================================================

def build_inputs(conversation_history):
    """Build model inputs (system prompt + full session history) on the
    model's device. Shared by the batch and streaming generation paths so
    both produce identical prompts."""
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                }
            ],
        }
    ]

    # Only replay the most recent turns so context (and latency) stays bounded
    # on long calls. The system prompt above is always kept in full.
    messages.extend(conversation_history[-MAX_HISTORY_MESSAGES:])

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )

    input_device = next(model.parameters()).device

    inputs = {
        k: v.to(input_device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    return inputs


# ============================================================
# GEMMA GENERATION (BATCH)
# ============================================================

def generate_reply(conversation_history):

    inputs = build_inputs(conversation_history)

    with _generation_lock:
        with torch.inference_mode():

            output = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
            )

    input_length = inputs["input_ids"].shape[-1]

    response = processor.decode(
        output[0][input_length:],
        skip_special_tokens=True,
    ).strip()

    return response


# ============================================================
# GEMMA GENERATION (STREAMING)
# ============================================================

def stream_reply(conversation_history):
    """Yield generated text incrementally as the model produces tokens.

    model.generate() runs on a background thread feeding a
    TextIteratorStreamer; this generator yields each decoded text piece as
    soon as it is available, so the caller can start TTS on the first
    sentence long before generation finishes. Generation kwargs are identical
    to the batch path (greedy, max_new_tokens=120)."""

    inputs = build_inputs(conversation_history)

    streamer = TextIteratorStreamer(
        STREAM_TOKENIZER,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        streamer=streamer,
    )

    def _run_generation():
        # Hold the lock for the whole generation so a concurrent turn waits
        # instead of corrupting the shared model state.
        with _generation_lock:
            with torch.inference_mode():
                model.generate(**generation_kwargs)

    thread = Thread(target=_run_generation, name="gemma-generate", daemon=True)
    thread.start()

    for new_text in streamer:
        if new_text:
            yield new_text

    thread.join()


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "Gemma 4 E4B conversation server",
    }


# ============================================================
# CHAT (BATCH — unchanged external contract)
# ============================================================

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    session_id = request.session_id
    text = request.text.strip()

    if not text:
        return ChatResponse(
            session_id=session_id,
            reply="",
        )

    # Create memory for new call
    if session_id not in sessions:

        sessions[session_id] = []

    # Add caller message
    sessions[session_id].append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text,
                }
            ],
        }
    )

    print()
    print("=" * 60)
    print("SESSION:", session_id)
    print("CALLER:", text)

    # Generate response using entire session memory
    reply = generate_reply(
        sessions[session_id]
    )

    # Save Gemma response
    sessions[session_id].append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": reply,
                }
            ],
        }
    )

    print("GEMMA:", reply)
    print("=" * 60)

    return ChatResponse(
        session_id=session_id,
        reply=reply,
    )


# ============================================================
# CHAT (STREAMING — token-by-token plain text)
#
# Emits generated text incrementally as `text/plain` chunks. Session memory
# is preserved exactly like /chat: the caller message is appended before
# generation and the full assistant reply is appended once streaming ends.
# Only ONE model.generate() runs per request.
# ============================================================

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):

    session_id = request.session_id
    text = request.text.strip()

    if not text:
        return StreamingResponse(iter(()), media_type="text/plain")

    # Create memory for new call
    if session_id not in sessions:
        sessions[session_id] = []

    # Add caller message
    sessions[session_id].append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text,
                }
            ],
        }
    )

    print()
    print("=" * 60)
    print("SESSION (stream):", session_id)
    print("CALLER:", text)

    def token_stream():
        collected = []
        for piece in stream_reply(sessions[session_id]):
            collected.append(piece)
            yield piece

        full_reply = "".join(collected).strip()

        # Save Gemma response into session memory
        sessions[session_id].append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": full_reply,
                    }
                ],
            }
        )

        print("GEMMA (stream):", full_reply)
        print("=" * 60)

    return StreamingResponse(
        token_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# CLEAR SESSION
# ============================================================

@app.delete("/session/{session_id}")
def clear_session(session_id: str):

    if session_id in sessions:
        del sessions[session_id]

    return {
        "status": "cleared",
        "session_id": session_id,
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
    )
