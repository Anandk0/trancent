# Docker setup — Divya voice pipeline

## Step 1 — Create 2 MIG slices on the host (run as root)

```bash
# Check what GPU you have
nvidia-smi -L

# Create 2x 3g.71gb slices from the 141 GB GPU
sudo nvidia-smi mig -cgi 3g.71gb,3g.71gb -C

# Verify — you should see 2 MIG devices
nvidia-smi -L
# Output:
#   MIG 3g.71gb Device 0: ... (UUID: MIG-aaa...)
#   MIG 3g.71gb Device 1: ... (UUID: MIG-bbb...)
```

## Step 2 — Create .env with your MIG UUIDs

```bash
cd docker/
cp .env.example .env
# Edit .env and paste the UUIDs from nvidia-smi -L
```

## Step 3 — Build and start

```bash
# First time — builds all images and downloads model weights (~24 GB)
HF_TOKEN=hf_your_token docker compose up --build

# After first run (weights cached in the hf-cache volume)
docker compose up -d
```

## Step 4 — Verify all containers are up

```bash
docker compose ps
docker compose logs -f
```

## Ports

| Container | Port | Service |
|---|---|---|
| gemma | 8000 | Gemma 4 LLM |
| svara | 8095 | Svara TTS model |
| asr | 8001 | ASR |
| agent | 8002 | Agent API |
| agent | 8003 | Svara adapter |
| agent | 8080 | Call server |

## Stop / restart

```bash
docker compose down          # stop all
docker compose restart gemma # restart one service
docker compose logs svara -f # follow one service's logs
```

## Model weight cache

Weights live in the `hf-cache` Docker volume (~24 GB). They are shared
across all containers. To wipe and re-download:

```bash
docker volume rm docker_hf-cache
```
