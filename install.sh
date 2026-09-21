#!/usr/bin/env bash
# Install everything needed to run the Divya voice AI pipeline on a fresh server.
#
# Run this once after unpacking divya-code.zip:
#
#   unzip divya-code.zip
#   cd trancent
#   bash install.sh
#
# What it does:
#   1. Creates 2 conda envs: gemma4-agent (Python 3.10) and svara (Python 3.11)
#   2. Installs all Python packages for each service
#   3. Downloads / pre-warms HuggingFace model weights (requires GPU + HF token)
#      - Gemma-4-E4B-it  (~15 GB)  -> service 1: LLM
#      - indic-conformer-600m       (~2.4 GB) -> service 2: ASR
#      - kenpath/svara-tts-v1       (~6 GB)   -> service 3: TTS
#   4. Checks ffmpeg is present (needed by ASR and Svara)
#
# Environment variables you can set before running:
#   HF_TOKEN      HuggingFace token (required for Gemma - gated model)
#   SKIP_DOWNLOAD set to "1" to skip weight download (if already cached)
#   GEMMA_ENV     conda env name for Gemma+ASR+agent (default: gemma4-agent)
#   SVARA_ENV     conda env name for Svara TTS       (default: svara)
#
# After this completes:
#   bash start_all.sh       <- brings all 6 services up

set -uo pipefail

GEMMA_ENV="${GEMMA_ENV:-gemma4-agent}"
SVARA_ENV="${SVARA_ENV:-svara}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

hr()  { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }
ok()  { printf '  \033[32m✓\033[0m  %s\n' "$1"; }
inf() { printf '  \033[34m→\033[0m  %s\n' "$1"; }
err() { printf '  \033[31m✗\033[0m  %s\n' "$1"; }
die() { err "$1"; exit 1; }

# ---------------------------------------------------------------- conda

hr "CONDA"
if ! command -v conda >/dev/null 2>&1; then
    die "conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
fi
ok "conda $(conda --version)"
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---------------------------------------------------------------- GPU check

hr "GPU"
if nvidia-smi -L 2>/dev/null | head -1; then
    ok "GPU visible"
else
    err "no GPU detected — model servers will run on CPU and be very slow"
fi

# ---------------------------------------------------------------- ENV 1: gemma4-agent
# Used by: gemma_server.py  asr_api.py  agent_api.py  call_server.py  svara_tts_api.py

hr "ENV 1: $GEMMA_ENV  (Gemma LLM + ASR + agent)"

if conda env list | grep -qE "^${GEMMA_ENV}\s"; then
    ok "env '$GEMMA_ENV' already exists"
else
    inf "creating env '$GEMMA_ENV' with python=3.10 ..."
    conda create -n "$GEMMA_ENV" python=3.10 -y \
        || die "failed to create $GEMMA_ENV"
    ok "created"
fi

inf "installing packages into $GEMMA_ENV ..."
conda run -n "$GEMMA_ENV" pip install --quiet \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    && ok "torch (CUDA 12.4)"

conda run -n "$GEMMA_ENV" pip install --quiet \
    transformers==5.16.1 \
    accelerate \
    fastapi \
    uvicorn[standard] \
    pydantic \
    requests \
    numpy \
    soundfile \
    torchaudio \
    && ok "ML / serving packages"

# ffmpeg — needed by ASR (audio decode) and Svara (pcm→wav conversion)
if command -v ffmpeg >/dev/null 2>&1; then
    ok "ffmpeg already on PATH"
else
    inf "installing ffmpeg via conda ..."
    conda install -n "$GEMMA_ENV" -c conda-forge -y ffmpeg \
        && ok "ffmpeg installed" \
        || err "ffmpeg install failed — ASR will error on every request"
fi

# ---------------------------------------------------------------- ENV 2: svara
# Used by: svara-tts-inference/api/server.py  (vLLM, autoregressive TTS)

hr "ENV 2: $SVARA_ENV  (Svara vLLM TTS)"

if conda env list | grep -qE "^${SVARA_ENV}\s"; then
    ok "env '$SVARA_ENV' already exists"
else
    inf "creating env '$SVARA_ENV' with python=3.11 (vLLM needs >= 3.9) ..."
    conda create -n "$SVARA_ENV" python=3.11 -y \
        || die "failed to create $SVARA_ENV"
    ok "created"
fi

inf "installing packages into $SVARA_ENV ..."
conda run -n "$SVARA_ENV" pip install --quiet \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    && ok "torch (CUDA 12.4)"

conda run -n "$SVARA_ENV" pip install --quiet \
    vllm \
    snac \
    fastapi \
    uvicorn[standard] \
    soundfile \
    numpy \
    && ok "vLLM + SNAC + serving packages"

conda install -n "$SVARA_ENV" -c conda-forge -y ffmpeg >/dev/null 2>&1 \
    && ok "ffmpeg in svara env" \
    || err "ffmpeg install failed in svara env — synthesis will return 500"

# ---------------------------------------------------------------- Clone Svara inference code

hr "SVARA INFERENCE CODE"
SVARA_DIR="$ROOT/svara-tts-inference"
if [ -d "$SVARA_DIR" ]; then
    ok "svara-tts-inference already present"
else
    inf "cloning kenpath/svara-tts-v1 ..."
    git clone https://github.com/kenpath/svara-tts-v1.git "$SVARA_DIR" \
        || die "git clone failed — check network access"
    ok "cloned to $SVARA_DIR"
fi

# ---------------------------------------------------------------- MODEL DOWNLOADS

if [ "$SKIP_DOWNLOAD" = "1" ]; then
    hr "SKIP_DOWNLOAD=1 — skipping weight download"
else

    hr "MODEL WEIGHTS"

    # Gemma 4 E4B — gated, needs HF token
    if [ -z "${HF_TOKEN:-}" ]; then
        err "HF_TOKEN is not set."
        inf "Gemma-4-E4B-it is a gated model. Get a token from https://huggingface.co/settings/tokens"
        inf "then: export HF_TOKEN=hf_xxx && bash install.sh"
        err "Skipping Gemma weight download."
    else
        inf "downloading google/gemma-4-E4B-it (~15 GB) ..."
        conda run -n "$GEMMA_ENV" python - <<PYEOF
import os
os.environ["HF_TOKEN"] = "${HF_TOKEN}"
from transformers import AutoProcessor, AutoTokenizer
from transformers import AutoModelForMultimodalLM
import torch
print("  downloading processor / tokenizer ...")
AutoProcessor.from_pretrained("google/gemma-4-E4B-it", token=os.environ["HF_TOKEN"])
AutoTokenizer.from_pretrained("google/gemma-4-E4B-it", token=os.environ["HF_TOKEN"])
print("  downloading model weights ...")
AutoModelForMultimodalLM.from_pretrained(
    "google/gemma-4-E4B-it",
    torch_dtype=torch.bfloat16,
    token=os.environ["HF_TOKEN"],
)
print("  gemma done")
PYEOF
        ok "Gemma-4-E4B-it downloaded"
    fi

    # ASR model — public
    inf "downloading ai4bharat/indic-conformer-600m-multilingual (~2.4 GB) ..."
    conda run -n "$GEMMA_ENV" python - <<PYEOF
from transformers import AutoModel, AutoProcessor
print("  downloading ASR model ...")
AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual")
print("  asr done")
PYEOF
    ok "indic-conformer-600m downloaded"

    # Svara TTS model — pulled by vLLM at server startup, but we can warm the HF cache now
    inf "downloading kenpath/svara-tts-v1 (~6 GB) ..."
    conda run -n "$SVARA_ENV" python - <<PYEOF
from huggingface_hub import snapshot_download
import os
print("  pulling svara weights into HF cache ...")
snapshot_download("kenpath/svara-tts-v1")
print("  svara done")
PYEOF
    ok "svara-tts-v1 downloaded"

fi  # end SKIP_DOWNLOAD

# ---------------------------------------------------------------- Gemma4-agent dir

hr "GEMMA4-AGENT DIRECTORY"
GEMMA_SUBDIR="$ROOT/gemma4-agent"
if [ ! -d "$GEMMA_SUBDIR" ]; then
    inf "gemma4-agent/ not found — creating with gemma_server.py symlink"
    mkdir -p "$GEMMA_SUBDIR"
    ln -sf "$ROOT/gemma_server.py" "$GEMMA_SUBDIR/gemma_server.py"
    ok "created symlink"
else
    ok "gemma4-agent/ present"
fi

# ---------------------------------------------------------------- ASR dir

hr "ASR DIRECTORY"
ASR_DIR="$ROOT/asr-server"
if [ -f "$ASR_DIR/asr_api.py" ]; then
    ok "asr-server/ present"
else
    die "asr-server/asr_api.py not found — the zip may be incomplete"
fi

# ---------------------------------------------------------------- Summary

hr "DONE"
cat <<EOF
  Code and weights are ready. Start everything with:

    bash $ROOT/start_all.sh

  Services that will come up:
    8095  Svara TTS model  (vLLM, takes ~75s to load)
    8003  Svara adapter    (lightweight proxy)
    8000  Gemma LLM        (takes ~60s to load)
    8001  ASR              (takes ~30s to load)
    8002  Agent API
    8080  Call server

  To move to a different server, zip the code (not weights):
    bash $ROOT/pack.sh /path/to/destination/divya-code.zip

  Weights live in ~/.cache/huggingface/hub/ and do NOT need to be moved —
  they are re-downloaded from HuggingFace on the new server by this script.
  If you have a fast NFS volume, point HF_HOME at it instead:
    export HF_HOME=/fast-volume/hf-cache
    bash install.sh
EOF
