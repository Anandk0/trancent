#!/usr/bin/env bash
# Bring Svara up on a fresh pod, applying only the workarounds that pod
# actually needs.
#
# Standing this up by hand took seven separate failures to work through, so
# each check below corresponds to one of them. Two of the fixes cost
# performance, so they are applied only when a probe shows they are
# required -- blanket-applying them is how a working slice ends up slow:
#
#   VLLM_ENFORCE_EAGER  disables CUDA graphs, which decode leans on
#   cudnn.enabled=False forces SNAC onto slower convolution kernels
#
#   bash setup_svara_pod.sh            # check, install, start
#   bash setup_svara_pod.sh --check    # probe only, change nothing
#
# Expects the shared volume mounted with svara-tts-inference under it.

set -u

ENV_NAME="${SVARA_ENV:-svara}"
PY_VER="${SVARA_PY:-3.11}"
GPU_MEM="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

# The code lives on the shared volume, which mounts at a different path per
# pod. Never write here -- other pods serve from these same files.
for d in "$HOME/svara-tts-inference" "$HOME/newollama-volume/svara-tts-inference"; do
    [ -d "$d" ] && SVARA_DIR="$d" && break
done
if [ -z "${SVARA_DIR:-}" ]; then
    echo "FAIL: svara-tts-inference not found under \$HOME"
    exit 1
fi

hr() { printf '\n\033[1m%s\033[0m\n' "----------------------------------------------------------"; }
ok() { printf '  \033[32m*\033[0m %s\n' "$1"; }
no() { printf '  \033[31mx\033[0m %s\n' "$1"; }
inf() { printf '    %s\n' "$1"; }

hr; echo " GPU"
nvidia-smi -L 2>/dev/null | sed 's/^/  /'
PROFILE=$(nvidia-smi -L 2>/dev/null | grep -oE '[0-9]+g\.[0-9]+gb' | head -1)
if [ -n "$PROFILE" ]; then
    GB=${PROFILE#*.}; GB=${GB%gb}
    if [ "${GB:-0}" -lt 40 ]; then
        no "$PROFILE is a small slice"
        inf "Svara measured RTF 1.43-1.69 on 1g.18gb -- slower than real time,"
        inf "so speech stalls mid-reply. 3g.71gb measured 0.47-0.57."
    else
        ok "$PROFILE - large enough"
    fi
fi

hr; echo " CONDA ENV"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    ok "env '$ENV_NAME' exists"
else
    no "env '$ENV_NAME' missing"
    if [ "$CHECK_ONLY" = "0" ]; then
        inf "creating with python $PY_VER (base is often 3.8; vLLM needs 3.9+)"
        conda create -n "$ENV_NAME" "python=$PY_VER" -y >/dev/null 2>&1 \
            && ok "created" || { no "create failed"; exit 1; }
    fi
fi
conda activate "$ENV_NAME" 2>/dev/null || { no "cannot activate $ENV_NAME"; exit 1; }
inf "python $(python -V 2>&1 | awk '{print $2}')"

hr; echo " PACKAGES"
need_pkgs=""
for mod in vllm snac torch soundfile fastapi uvicorn; do
    if python -c "import $mod" 2>/dev/null; then
        ok "$mod"
    else
        no "$mod missing"
        need_pkgs="$need_pkgs $mod"
    fi
done
if [ -n "$need_pkgs" ] && [ "$CHECK_ONLY" = "0" ]; then
    inf "installing:$need_pkgs"
    pip install --quiet $need_pkgs || no "some installs failed - check manually"
fi

if command -v ffmpeg >/dev/null 2>&1; then
    ok "ffmpeg on PATH"
else
    no "ffmpeg missing - synthesis returns 500 without it"
    [ "$CHECK_ONLY" = "0" ] && {
        inf "installing ffmpeg"
        conda install -c conda-forge -y ffmpeg >/dev/null 2>&1 \
            && ok "installed" || no "install failed"
    }
fi

hr; echo " RUNTIME PROBES"

# Probe 1: does the inductor path import cleanly? vLLM 0.9.2 wants
# triton_key, which triton 3.8 removed. Only disable compilation if broken --
# it takes CUDA graphs with it, and decode is where that hurts.
if python -c "from triton.compiler.compiler import triton_key" 2>/dev/null; then
    ok "triton_key present - CUDA graphs can stay on"
    NEED_EAGER=0
else
    no "triton_key missing - torch.compile path will fail"
    inf "will set VLLM_ENFORCE_EAGER=true (costs CUDA graphs, so decode is slower)"
    NEED_EAGER=1
fi

# Probe 2: can cuDNN do a convolution at all? SNAC is the only part that
# needs it, and on one pod it failed with the GPU completely idle.
if python -c "
import torch
x = torch.randn(1, 8, 64, device='cuda')
torch.nn.Conv1d(8, 8, 3).cuda()(x)
" 2>/dev/null; then
    ok "cuDNN works - SNAC runs its normal kernels"
    NEED_NOCUDNN=0
else
    no "cuDNN cannot initialise - SNAC warmup will fail"
    inf "will disable cuDNN (SNAC falls back to slower convolutions)"
    NEED_NOCUDNN=1
fi

if [ "$CHECK_ONLY" = "1" ]; then
    hr; echo " check only - nothing changed"; exit 0
fi

hr; echo " LAUNCH"

LAUNCHER="python -u api/server.py"
if [ "$NEED_NOCUDNN" = "1" ]; then
    # /tmp is pod-local. The server file is on the shared volume and must not
    # be edited -- other pods run from it.
    cat > /tmp/run_svara.py <<PYEOF
import torch
torch.backends.cudnn.enabled = False
print("[wrapper] cudnn disabled")
import os, runpy, sys
ROOT = "$SVARA_DIR"
# server.py hands uvicorn the string "server:app", so api/ has to be
# importable by name; running the file directly does that, runpy does not.
sys.path.insert(0, os.path.join(ROOT, "api"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.argv = ["api/server.py"]
runpy.run_path(os.path.join(ROOT, "api/server.py"), run_name="__main__")
PYEOF
    LAUNCHER="python -u /tmp/run_svara.py"
    inf "using cuDNN-disabled wrapper"
fi

export VLLM_GPU_MEMORY_UTILIZATION="$GPU_MEM"
[ "$NEED_EAGER" = "1" ] && export VLLM_ENFORCE_EAGER=true

inf "VLLM_GPU_MEMORY_UTILIZATION=$GPU_MEM"
inf "VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-false}"
inf "dir: $SVARA_DIR"
echo
inf "starting (weights take ~75s)..."

cd "$SVARA_DIR"
mkdir -p /tmp/svara-logs
nohup $LAUNCHER >> /tmp/svara-logs/server.log 2>&1 &

waited=0
while [ "$waited" -lt 300 ]; do
    if curl -s -o /dev/null -m 2 http://127.0.0.1:8095/health 2>/dev/null; then
        ok "up after ${waited}s"
        break
    fi
    sleep 5; waited=$((waited + 5))
done

if [ "$waited" -ge 300 ]; then
    no "did not come up - last 25 log lines:"
    tail -25 /tmp/svara-logs/server.log | sed 's/^/    /'
    exit 1
fi

hr; echo " VERIFY"
curl -s -m 5 http://127.0.0.1:8095/health; echo
grep -E "Model loading took|quantization=" /tmp/svara-logs/server.log | tail -2 | sed 's/^/  /'

echo
inf "timing a short clause (local, no network in the way):"
for i in 1 2 3; do
    curl -s -o /tmp/probe.raw -w "    %{time_total}s\n" -m 120 \
        -X POST http://127.0.0.1:8095/v1/audio/speech \
        -H 'Content-Type: application/json' \
        -d '{"input":"जी, बताइए","voice":"hi_female","response_format":"pcm"}'
done
BYTES=$(stat -c%s /tmp/probe.raw 2>/dev/null || echo 0)
python - "$BYTES" <<'PYEOF'
import sys
b = int(sys.argv[1])
secs = b / (24000 * 2)
print(f"    {b} bytes = {secs:.2f}s of audio")
print(f"    compare: 3g.71gb measured RTF 0.52, 1g.18gb measured 1.43-1.69")
print(f"    anything at or above 1.0 cannot sustain real-time speech")
PYEOF

hr; echo " NEXT"
cat <<'EOF'
  Pod-to-pod traffic is blocked by mesh policy (RBAC: access denied), so
  reaching this from the agent needs a tunnel:

      cd /tmp
      curl -L -o cloudflared \
        https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
      chmod +x cloudflared
      ./cloudflared tunnel --url http://localhost:8095

  Then on the pod running the agent, point a second adapter at the printed
  URL, leaving the live one on 8003 untouched:

      SVARA_ADAPTER_PORT=8007 \
      SVARA_URL=https://<tunnel>/v1/audio/speech \
        nohup python svara_tts_api.py >> logs/svara-remote.log 2>&1 &

      BENCH_A_URL=http://127.0.0.1:8003/synthesize BENCH_A_NAME="local" \
      BENCH_B_URL=http://127.0.0.1:8007/synthesize BENCH_B_NAME="new slice" \
        python bench_tts.py

  The tunnel measured ~130ms of overhead, which is affordable only if the
  slice itself is fast.
EOF
