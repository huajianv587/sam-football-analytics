# PitchVision Setup

PitchVision runs the compact English web UI and FastAPI controller on the local
machine, stores business data and final artifacts in Supabase, and submits one
offline Slurm job to an NVIDIA A40. Passwords, API secrets, footage, model
weights and generated artifacts never belong in Git.

## 1. Prerequisites

- Node.js 20 or newer
- Python 3.11 or 3.12 and [`uv`](https://docs.astral.sh/uv/)
- FFmpeg, OpenSSH and `rsync`
- A Supabase project
- Passwordless SSH access to a Slurm cluster with one CUDA GPU

## 2. Local environment

Create separate frontend and backend environment files:

```bash
cp .env.example web/.env.local
cp .env.example backend/.env
```

Only these values may be exposed to the browser bundle:

```dotenv
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=
NEXT_PUBLIC_INFERENCE_API_URL=http://127.0.0.1:8000
```

The backend additionally requires:

```dotenv
SUPABASE_URL=
SUPABASE_PUBLISHABLE_KEY=
SUPABASE_SECRET_KEY=
LOCAL_ADMIN_USER_ID=
AUTH_DISABLED=true
ALLOWED_ORIGINS=http://localhost:3000
TC2_HOST=
TC2_USER=
TC2_PORT=22
TC2_REMOTE_ROOT=/scratch-share/<username>/sam-football
```

`AUTH_DISABLED=true` is the local single-admin mode used by the portfolio demo:
opening `/` immediately shows the upload workspace and `/login` redirects home.
FastAPI still stamps every project and Storage path with `LOCAL_ADMIN_USER_ID`.
Set `AUTH_DISABLED=false` before exposing the service to multiple users.

Never prefix `SUPABASE_SECRET_KEY` with `NEXT_PUBLIC_`.

## 3. Supabase

Create a project near the application users and execute the migrations in order:

1. `supabase/migrations/001_initial.sql`
2. `supabase/migrations/002_seed_roster.sql`
3. `supabase/migrations/003_service_role_grants.sql`
4. `supabase/migrations/004_auto_tracking.sql`
5. `supabase/migrations/005_verify_2026_final_roster.sql`

They create private `videos` and `artifacts` buckets, owner-scoped RLS policies,
`projects`, `tracks` and `roster`, per-track mask paths, automatic/manual identity
fields, calibration paths, stage/progress fields, and the verified 2026 final
rosters. The secret key is used only by FastAPI.

For the local demo, set `LOCAL_ADMIN_USER_ID` to an existing Supabase Auth user
UUID. Browser sign-in and public registration are not part of this flow.

## 4. Remote GPU runtime

Bootstrap the two pinned remote environments into writable scratch storage:

```bash
bash backend/scripts/bootstrap_remote.sh \
  /scratch-share/<username>/sam-football/runtime
```

The script creates:

- a Python 3.11 SAM environment with PyTorch/CUDA, EasyOCR, FFmpeg and the
  compiled SAM 2 CUDA extension;
- a separate SoccerNet/SoccerMaster game-state environment for football YOLO,
  PRTReID, StrongSORT and PnLCalib;
- version-pinned checkpoints and checksum manifests under scratch.

The default profile does not install the optional Qwen role model. To prepare
that experimental path explicitly:

```bash
INSTALL_ROLE_MODEL=true bash backend/scripts/bootstrap_remote.sh \
  /scratch-share/<username>/sam-football/runtime
```

Review [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) before redistributing
the runtime or using it commercially.

The checked-in `backend/scripts/job.sbatch` profile requests the authorized TC2
`normal` QoS, one A40, 10 CPUs, 30 GB system RAM and a two-hour limit. Adjust the
partition and resources for another cluster.

### Fast profile

The default job profile is the measured portfolio configuration:

```text
GSR_ANALYSIS_FPS=5
GSR_COMBINED=true
GSR_CONCAT_TRACKLETS=false
SAM_OBJECT_BATCH=64  # measured acceptance override: 96 for 71 Tracks on A40
SAM_BIDIRECTIONAL=false
SAM_PROMPTS_PER_TRACK=2
SAM_VOS_OPTIMIZED=false
ROLE_MODEL_ENABLED=false
```

It reconstructs game state at 5 FPS, maps observations back to the 15 FPS SAM
timeline, runs detection/ReID/tracking/calibration in one TrackLab process, and
uses one forward SAM propagation anchored by the first detection plus one
high-quality later prompt. Tracks shorter than two seconds are treated as
fragments, not unique analysis subjects.

For a slower quality-comparison experiment, set `SAM_BIDIRECTIONAL=true` and
increase `SAM_PROMPTS_PER_TRACK`. Do not present that profile as the measured
default unless it is benchmarked separately.

## 5. Start local services

Backend:

```bash
cd backend
uv sync --extra test
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend, in another terminal:

```bash
cd web
npm ci
npm run dev
```

Open `http://localhost:3000`. The controller can be checked independently:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok","scheduler":"tc2-slurm"}
```

## 6. Run an analysis

1. Open the upload workspace; no login or annotation screen appears.
2. Select an H.264 MP4, maximum 50 MB and 60 seconds.
3. The browser submits the video directly to FastAPI.
4. Observe Normalize, Detect/Track/Calibrate, Segment, Identify and Upload.
5. Open the completed result. All currently visible subjects initially show a
   thin box and Track ID.
6. Click one box to lazy-load that track's cropped RLE Mask, trajectory, speed,
   distance, team, role, identity confidence and occlusion metrics.
7. If automatic jersey OCR is unavailable, select a verified roster player from
   the correction menu; the manual override persists in Supabase.

Metric speed is hidden when automatic calibration is invalid. The UI never
substitutes a fabricated `0 km/h`.

## 7. Verification

```bash
backend/.venv/bin/python -m pytest -q
cd web
npm test -- --run
npm run lint
npm run build
```

Validate a returned GPU result bundle separately:

```bash
PYTHONPATH=backend backend/.venv/bin/python \
  backend/scripts/validate_artifacts.py /path/to/results
```

This decodes every per-track Mask, checks Mask/box consistency, parses every
calibration frame, counts roles/teams/metric tracks and fully decodes both MP4s.

## 8. Scope

- Offline MP4 only; RTSP/HLS/WebRTC, browser cameras and drone feeds are roadmap
  input adapters.
- Continuous broadcast shots only; camera cuts deliberately create new IDs.
- Automatic OCR returns `Unidentified` when the number is not genuinely visible.
- The repository contains no source footage, generated media, checkpoints or
  third-party runtime trees.
