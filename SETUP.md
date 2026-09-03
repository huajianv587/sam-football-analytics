# PitchVision Setup

PitchVision runs the compact English web UI and FastAPI controller on the local
machine. Offline analysis stores final artifacts in Supabase; live analysis uses
a six-hour A40 WebSocket worker through an SSH tunnel. Passwords, API secrets,
footage, model weights and generated artifacts never belong in Git.

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
NEXT_PUBLIC_LIVE_WS_URL=ws://127.0.0.1:8010/v1/live/ws
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

# Optional live object classes (COCO labels)
LIVE_CLASSES=person,cat,dog,bird,horse,sheep,cow,chair,couch,bed,dining table
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
6. `supabase/migrations/006_generic_upload_defaults.sql`

They create private `videos` and `artifacts` buckets, owner-scoped RLS policies,
`projects`, `tracks` and `roster`, per-track mask paths, automatic/manual identity
fields, calibration paths, stage/progress fields, and the verified 2026 final
rosters. The final migration makes new arbitrary uploads default to generic
teams without changing existing projects or roster rows. The secret key is used
only by FastAPI.

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
- Ultralytics `8.4.138`, `yolo11s-seg.pt`, FastAPI and Uvicorn for generic
  all-person live instance segmentation;
- a separate SoccerNet/SoccerMaster game-state environment for football YOLO
  and PnLCalib, plus legacy PRTReID/StrongSORT compatibility components;
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

The separate `backend/scripts/live.sbatch` requests the same CPU/GPU/memory
profile for up to six hours and starts the live worker on the compute node. It
binds only inside the cluster network; the browser reaches it through the SSH
tunnel below.

### Default quality/performance profile

The checked-in job profile separates model sampling rates and fixes the SAM
compile shape:

```text
DETECTOR_TRACKER_FPS=10
REID_APPEARANCE_FPS=5
CALIBRATION_FPS=5
GSR_CONCAT_TRACKLETS=false
SAM_WINDOW_FRAMES=90
SAM_WINDOW_OVERLAP=15
SAM_OBJECT_BATCH=16
SAM_BIDIRECTIONAL=true
SAM_PROMPTS_PER_TRACK=3
SAM_VOS_OPTIMIZED=true
SAM_PAD_COMPILED_BATCH=true
ROLE_MODEL_ENABLED=false
```

The football detector observes at 10 FPS. A low-cost HSV appearance descriptor
is sampled at 5 FPS and contributes only five percent of association score.
PnLCalib runs independently at 5 FPS and its homographies are temporally
interpolated to the 15 FPS output timeline. High-confidence detections create
tracks; low-confidence detections can recover an existing track but never create
one.

The default all-person model is `sam2.1_hiera_base_plus.pt`. It processes only
tracks whose lifetimes intersect a 90-frame window. The final object bucket is
padded to 16 only inside the compiled predictor, so dynamic object counts do not
cause a new graph compile; dummy identities are filtered before Mask export.
Clicking **REFINE SELECTED WITH SAM LARGE** submits a separate one-track job and
atomically replaces that track's Storage object when the Large Mask succeeds.

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

## 6. Start live person segmentation

Submit the dedicated A40 worker. This syncs only backend source code into the
configured scratch root and returns a Slurm Job ID:

```bash
bash backend/scripts/submit_live.sh
```

After `squeue` reports the job as running, keep this tunnel open in a separate
terminal:

```bash
bash backend/scripts/live_tunnel.sh <job-id>
```

Verify the worker without loading the models:

```bash
curl http://127.0.0.1:8010/health
```

Open `http://localhost:3000/live` and choose a video or field camera. The first
WebSocket session loads YOLO11 Segment and SAM Base+, so its initial ready state
is slower than later connections.

- `ALL MASKS`: draw every generic person instance Mask.
- `SELECTED ONLY`: draw only the selected Track Mask.
- `BOXES`: draw Track boxes and IDs without Mask fills.
- Clicking a Track switches its Mask source from `lightweight` to `sam` on the
  next processed frame. Clearing selection returns to all-lightweight mode.

The browser keeps one exact frame in flight; if inference is slower than capture,
it samples a newer frame instead of queuing old frames. Every response is drawn
over the matching stored `ImageBitmap`, which trades a bounded amount of latency
for correct visual alignment.

Generic tracking always provides pixel speed. Add a validated sport-specific
Homography before displaying km/h; the live page intentionally shows
`Calibration required` instead of guessing physical units.

## 7. Run an offline analysis

1. Open the upload workspace; no login or annotation screen appears.
2. Select an H.264 MP4, maximum 50 MB and 60 seconds.
3. The browser submits the video directly to FastAPI.
4. Observe Normalize, Detect/Track/Calibrate, Segment, Identify and Upload.
5. Open the completed result. All currently visible subjects initially show a
   thin box and Track ID.
6. Click one box to lazy-load that track's cropped RLE Mask, trajectory, speed,
   distance, team, role, identity confidence and occlusion metrics.
   Directly verified Mask frames remain the stored source of truth. During
   playback, an isolated quality-gated gap is bridged visually by projecting the
   nearest verified contour onto the current detector box, while the reported
   direct Mask coverage remains unchanged.
7. If automatic jersey OCR is unavailable, select a verified roster player from
   the correction menu; the manual override persists in Supabase.

The direct upload path defaults to `Unspecified Match`, `Team A` and `Team B`.
An empty roster is valid: the person remains `Unidentified`, while its Track ID,
Mask, trajectory, occlusion summary and calibrated movement metrics remain
available. For a known fixture, populate `roster` and send the matching fixture
and team labels through the API.

Metric speed is hidden when automatic calibration is invalid. The UI never
substitutes a fabricated `0 km/h`.

## 8. Verification

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

Structural validation is not ground-truth evaluation. To compute detection
precision/recall, global IDF1, human-labelled Mask IoU and calibration error,
copy `validation/annotation-template.json`, label the requested frames and run:

```bash
PYTHONPATH=backend backend/.venv/bin/python \
  backend/scripts/evaluate_ground_truth.py \
  /path/to/results /path/to/annotations.json
```

## 9. Scope

- Live browser video and camera input plus offline MP4 are implemented.
- Native GPU-side RTSP/HLS ingest, WebRTC media output and drone transport are
  roadmap adapters over the live result protocol.
- Continuous broadcast shots only; camera cuts deliberately create new IDs.
- Automatic OCR returns `Unidentified` when the number is not genuinely visible.
- The repository contains no source footage, generated media, checkpoints or
  third-party runtime trees.
