# Setup Guide

This guide keeps the local web/API process separate from the remote GPU runtime. No SSH password, Supabase secret key, source video, or model checkpoint belongs in Git.

## 1. Prerequisites

- Node.js 20+
- Python 3.11 or 3.12 and [`uv`](https://docs.astral.sh/uv/)
- FFmpeg and `rsync`
- A Supabase project
- Passwordless SSH access to a Slurm cluster with one CUDA GPU

## 2. Environment files

Copy the example values into separate frontend and backend files:

```bash
cp .env.example web/.env.local
cp .env.example backend/.env
```

The browser bundle may contain only:

```dotenv
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=
NEXT_PUBLIC_INFERENCE_API_URL=http://127.0.0.1:8000
```

The FastAPI environment additionally needs:

```dotenv
SUPABASE_URL=
SUPABASE_PUBLISHABLE_KEY=
SUPABASE_SECRET_KEY=
LOCAL_ADMIN_USER_ID=
AUTH_DISABLED=false
ALLOWED_ORIGINS=http://localhost:3000
TC2_HOST=
TC2_USER=
TC2_PORT=22
TC2_REMOTE_ROOT=/scratch-share/<username>/sam-football
```

`SUPABASE_SECRET_KEY` is backend-only. Never prefix it with `NEXT_PUBLIC_`.

## 3. Supabase

Create a project, preferably near the application users, then run the migrations in order from the Supabase SQL Editor:

1. `supabase/migrations/001_initial.sql`
2. `supabase/migrations/002_seed_roster.sql`
3. `supabase/migrations/003_service_role_grants.sql`

The migrations create:

- `projects`, `tracks`, and `roster` tables;
- private `videos` and `artifacts` Storage buckets;
- owner-scoped Row Level Security policies;
- service-role grants needed by the FastAPI controller.

Disable public sign-up for the single-admin demo and create or invite the administrator account manually. Put its UUID in `LOCAL_ADMIN_USER_ID` for the simplified offline endpoint.

## 4. Remote A40 runtime

The API uses the system `ssh` and `rsync` clients and the user's existing SSH key. It never stores an SSH password.

On the cluster, choose a writable runtime directory and run:

```bash
bash backend/scripts/bootstrap_remote.sh /scratch-share/<username>/sam-football/runtime
```

The script creates an isolated Python 3.11 environment, installs PyTorch for CUDA 12.8, compiles the SAM 2 CUDA extension for Ampere (`sm_86`), downloads `sam2.1_hiera_large.pt`, and redirects Torch, Conda, pip, EasyOCR, Triton, and TorchInductor caches into scratch storage.

Adjust the `#SBATCH` partition, QoS, CPU, memory, and time directives in `backend/scripts/job.sbatch` for another cluster. The current profile requests one GPU, eight CPUs, 24 GB RAM, and a two-hour limit.

## 5. Local services

Backend:

```bash
cd backend
uv sync --extra test
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd web
npm ci
npm run dev
```

Open `http://localhost:3000` and verify the controller independently:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok","scheduler":"tc2-slurm"}
```

## 6. Test and build

```bash
backend/.venv/bin/python -m pytest -q
cd web && npm test && npm run build
```

The full GPU acceptance test uses a continuous 30-second, 1280x720, 15 FPS clip with 11 first-frame prompts. Local/copyrighted footage is intentionally excluded from this repository.

## 7. Input contract

- MP4 / H.264
- 50 MB maximum in the demo API
- 60 seconds maximum
- one continuous camera shot
- one first-frame box per tracked subject
- four video-to-pitch calibration pairs for metric speed

Version 1 does not perform cross-shot ReID. A subject that leaves the image is finalized at the frame boundary; a similar-looking player is not allowed to inherit that persistent ID.
