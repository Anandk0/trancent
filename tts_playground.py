"""
TTS playground - type text, hear it, see what it cost.

Comparing TTS placements by curl means decoding base64 by hand and never
actually listening to the result. This serves a page that speaks whatever
you type, against whichever backend you pick, and reports latency and
real-time factor next to the audio.

RTF is the number that decides whether a placement is usable: above 1.0 the
engine produces speech slower than it plays, so a reply stalls mid-sentence
no matter how good the first-clause latency looks.

    python tts_playground.py            # http://127.0.0.1:8011

Backends default to the local adapter (8003) and the remote one (8007).
Override:

    TTS_BACKENDS="local=http://127.0.0.1:8003/synthesize,gpu2=http://127.0.0.1:8007/synthesize" \
        python tts_playground.py
"""

import base64
import io
import os
import time

from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

HOST = os.environ.get("TTS_PLAYGROUND_HOST", "0.0.0.0")
PORT = int(os.environ.get("TTS_PLAYGROUND_PORT", "8011"))

_DEFAULT_BACKENDS = (
    "local (8003)=http://127.0.0.1:8003/synthesize,"
    "remote (8007)=http://127.0.0.1:8007/synthesize"
)


def _parse_backends(raw):
    out = {}
    for pair in raw.split(","):
        if "=" in pair:
            name, url = pair.split("=", 1)
            out[name.strip()] = url.strip()
    return out


BACKENDS = _parse_backends(os.environ.get("TTS_BACKENDS", _DEFAULT_BACKENDS))

http_session = requests.Session()
http_session.mount(
    "http://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8)
)

app = FastAPI(title="TTS playground")


class SpeakRequest(BaseModel):
    text: str
    # Optional[...] rather than `str = None`: the page sends an explicit null
    # for a blank voice box, and a bare `str` default rejects that as a type
    # error before the handler ever runs.
    backend: Optional[str] = None
    voice: Optional[str] = None


def _audio_seconds(audio_b64, sample_rate):
    try:
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(base64.b64decode(audio_b64)))
        return len(data) / sr
    except Exception:
        # 16-bit mono PCM behind a 44-byte header
        return max(len(base64.b64decode(audio_b64)) - 44, 0) / (2 * sample_rate)


@app.get("/backends")
def backends():
    return {"backends": list(BACKENDS.keys())}


@app.post("/speak")
def speak(req: SpeakRequest):
    text = (req.text or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "text is empty"})

    name = req.backend or next(iter(BACKENDS))
    url = BACKENDS.get(name)
    if not url:
        return JSONResponse(status_code=400, content={"error": f"unknown backend {name}"})

    payload = {"text": text}
    if req.voice:
        payload["voice"] = req.voice

    t0 = time.perf_counter()
    try:
        r = http_session.post(url, json=payload, timeout=180)
        r.raise_for_status()
        j = r.json()
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"error": f"{type(e).__name__}: {e}", "backend": name, "url": url},
        )
    elapsed = time.perf_counter() - t0

    b64 = j.get("audio_b64")
    if not b64:
        return JSONResponse(
            status_code=502,
            content={"error": "backend returned no audio", "response": str(j)[:400]},
        )

    sr = j.get("sample_rate", 24000)
    dur = _audio_seconds(b64, sr)
    print(
        f"[PLAYGROUND] {name:<14} {elapsed:6.3f}s  audio {dur:5.2f}s  "
        f"RTF {elapsed / dur if dur else float('nan'):5.3f}  {text[:48]}"
    )

    return {
        "audio_b64": b64,
        "sample_rate": sr,
        "voice": j.get("voice"),
        "backend": name,
        "latency_s": round(elapsed, 3),
        "audio_s": round(dur, 2),
        "rtf": round(elapsed / dur, 3) if dur else None,
        "kb": round(len(b64) / 1024),
    }


PAGE = """
<title>TTS playground</title>
<style>
  :root {
    --bg:#f4f6f8; --card:#fff; --ink:#131a24; --muted:#5a6675;
    --line:#dce2ea; --accent:#a8690b; --good:#15803d; --bad:#9c2c22;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#121822; --card:#1a2230; --ink:#e3e9f1; --muted:#94a2b6;
            --line:#29333f; --accent:#e0a340; --good:#55c57f; --bad:#e68c80; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.6 'IBM Plex Sans',system-ui,-apple-system,sans-serif; }
  .wrap { max-width:820px; margin:0 auto; padding:28px 20px 60px; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:var(--muted); margin:0 0 22px; font-size:14px; }
  .card { background:var(--card); border:1px solid var(--line);
          border-radius:6px; padding:18px; margin-bottom:18px; }
  label { display:block; font-size:11px; letter-spacing:.08em;
          text-transform:uppercase; color:var(--muted); margin-bottom:6px; }
  textarea, input, select, button {
    font:inherit; color:inherit; background:var(--bg);
    border:1px solid var(--line); border-radius:4px; padding:9px 11px; width:100%; }
  textarea { min-height:90px; resize:vertical; }
  .row { display:flex; gap:12px; flex-wrap:wrap; margin-top:14px; }
  .row > div { flex:1 1 160px; }
  button { background:var(--accent); color:#fff; border:none; cursor:pointer;
           font-weight:600; padding:11px 18px; }
  button:disabled { opacity:.55; cursor:default; }
  button:focus-visible { outline:2px solid var(--ink); outline-offset:2px; }
  .stats { display:flex; gap:22px; flex-wrap:wrap; margin-top:4px;
           font-family:'IBM Plex Mono',ui-monospace,monospace; font-size:13px; }
  .stats b { display:block; font-size:19px; font-variant-numeric:tabular-nums; }
  .stats span { color:var(--muted); font-size:11px; letter-spacing:.06em;
                text-transform:uppercase; }
  .rtf-ok { color:var(--good); } .rtf-bad { color:var(--bad); }
  audio { width:100%; margin-top:14px; }
  .err { color:var(--bad); font-family:ui-monospace,monospace; font-size:13px;
         white-space:pre-wrap; }
  table { width:100%; border-collapse:collapse; font-size:13px;
          font-family:'IBM Plex Mono',ui-monospace,monospace; }
  th { text-align:left; font-size:10px; letter-spacing:.08em; text-transform:uppercase;
       color:var(--muted); border-bottom:1px solid var(--line); padding:0 10px 6px 0; }
  td { padding:7px 10px 7px 0; border-bottom:1px solid var(--line);
       font-variant-numeric:tabular-nums; }
  td.txt { font-family:'IBM Plex Sans',sans-serif; color:var(--muted); }
  .hint { color:var(--muted); font-size:12.5px; margin-top:10px; }
</style>

<div class="wrap">
  <h1>TTS playground</h1>
  <p class="sub">Type something, hear it, and see what it cost.</p>

  <div class="card">
    <label for="text">Text</label>
    <textarea id="text">Hello, this is a test of the speech system. Can you hear me clearly?</textarea>

    <div class="row">
      <div>
        <label for="backend">Backend</label>
        <select id="backend"></select>
      </div>
      <div>
        <label for="voice">Voice (optional)</label>
        <input id="voice" placeholder="leave blank for default">
      </div>
      <div style="flex:0 0 auto; display:flex; align-items:flex-end;">
        <button id="speak">Speak</button>
      </div>
    </div>
    <p class="hint">Tip: paste a long sentence. Short clips hide the problem —
       real-time factor is what tells you whether speech will stall mid-reply.</p>
  </div>

  <div class="card" id="result" hidden>
    <div class="stats">
      <div><span>latency</span><b id="s-lat">—</b></div>
      <div><span>audio length</span><b id="s-dur">—</b></div>
      <div><span>real-time factor</span><b id="s-rtf">—</b></div>
      <div><span>size</span><b id="s-kb">—</b></div>
    </div>
    <audio id="player" controls></audio>
    <p class="hint" id="rtf-note"></p>
  </div>

  <div class="card" id="errcard" hidden>
    <label>Error</label>
    <div class="err" id="err"></div>
  </div>

  <div class="card">
    <label>History</label>
    <table>
      <thead><tr><th>backend</th><th>latency</th><th>audio</th><th>RTF</th><th>text</th></tr></thead>
      <tbody id="hist"></tbody>
    </table>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);

fetch('backends').then(r => r.json()).then(j => {
  $('backend').innerHTML = j.backends.map(b => `<option>${b}</option>`).join('');
});

async function speak() {
  const text = $('text').value.trim();
  if (!text) return;

  $('speak').disabled = true;
  $('speak').textContent = 'Speaking…';
  $('errcard').hidden = true;

  try {
    const r = await fetch('speak', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        text,
        backend: $('backend').value,
        voice: $('voice').value.trim() || null
      })
    });
    const j = await r.json();

    if (!r.ok) {
      $('err').textContent = j.error + (j.url ? '\\n' + j.url : '');
      $('errcard').hidden = false;
      return;
    }

    $('s-lat').textContent = j.latency_s + 's';
    $('s-dur').textContent = j.audio_s + 's';
    $('s-kb').textContent  = j.kb + ' KB';

    const rtf = $('s-rtf');
    rtf.textContent = j.rtf;
    rtf.className = j.rtf < 1 ? 'rtf-ok' : 'rtf-bad';
    $('rtf-note').textContent = j.rtf < 1
      ? 'Under 1.0 — generates faster than it plays, so speech runs smoothly.'
      : 'Above 1.0 — generates slower than it plays. Longer replies will stall mid-sentence.';

    $('player').src = 'data:audio/wav;base64,' + j.audio_b64;
    $('result').hidden = false;
    $('player').play().catch(() => {});

    const row = document.createElement('tr');
    row.innerHTML = `<td>${j.backend}</td><td>${j.latency_s}s</td>` +
                    `<td>${j.audio_s}s</td>` +
                    `<td class="${j.rtf < 1 ? 'rtf-ok' : 'rtf-bad'}">${j.rtf}</td>` +
                    `<td class="txt">${text.slice(0, 40)}</td>`;
    $('hist').prepend(row);
  } catch (e) {
    $('err').textContent = String(e);
    $('errcard').hidden = false;
  } finally {
    $('speak').disabled = false;
    $('speak').textContent = 'Speak';
  }
}

$('speak').addEventListener('click', speak);
$('text').addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') speak();
});
</script>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    print(f"TTS playground on http://{HOST}:{PORT}")
    for n, u in BACKENDS.items():
        print(f"  {n:<16} -> {u}")
    uvicorn.run(app, host=HOST, port=PORT)
