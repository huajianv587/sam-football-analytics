# PitchVision quantified capability review

All values in this document are rounded to two decimal places. “Measured”
means it was observed on the supplied 1280 x 720, 15 FPS broadcast through the
active NVIDIA A40 service. “Configured” describes a model or software limit and
is not an accuracy claim.

## Latest reproducible acceptance run (2026-09-03)

The current live-worker build was deployed to TC2 A40 job `33522` and tested
against `西班牙_阿根廷_连续镜头_30s_720p15.mp4`. This is an indexed video
precompute run, so every reported frame is retained for native browser
playback, pause and seek rather than being recomputed after a seek.

| Quantity | Value | Meaning |
| --- | ---: | --- |
| Source duration / frames | 30.00 s / 450.00 | 1280 x 720 at 15.00 FPS |
| Returned frames with people | 450.00 / 450.00 | No empty-frame gap in this run |
| Visible people per frame | 11.00 / 18.00 / 19.00 | Minimum / median / maximum |
| Distinct public Track IDs | 29.00 | Monotonic IDs across the complete session |
| Lightweight inference P50 / P95 | 62.10 / 64.20 ms | A40 service measurement |
| Mean reported processing rate | 15.88 FPS | Service-side throughput |
| Predicted Track observations | 1,427.00 | Explicit short-gap compensation, not detector facts |
| Reported public-ID reassociations | 0.00 | Stable registry counter for this run |
| Indexed seek round trip | Exact | Frame 0 response before and after a seek to frame 210 matched byte-for-byte |
| Selected SAM refinement | 31.00 frames | Track 1, centre frame 15, radius 15, state `ready` |
| Mac failover smoke | 15.00 / 15.00 frames | A forced A40 connection failure completed using the local MPS worker |
| Browser upload and precompute | Passed | The visible `VIDEO FILE` control submitted the 30.00-second source and reached `READY` |
| Browser selected-track refinement | Passed | Clicking Track 1 changed `SAM REFINING` to `SAM READY` |
| Browser playback controls | Passed | `PLAY`, `PAUSE`, `-5s`, `+5s`, `RESTART`, and native controls enabled only after indexing |
| Native seek round trip | Passed | Dragged 1.27 s -> 17.94 s -> 1.27 s while preserving Track 1 selection and visible Tracks |
| Failure retry | Passed | Both workers disabled -> `FAILED` -> `RETRY ANALYSIS`; A40 restored -> same file reached `READY` |

The A40 route is selected whenever it is healthy. Mac MPS is a deliberate
fallback for lightweight processing only; its UI and API response identify the
executor, and selected-person SAM reports its A40 requirement instead of
claiming a refinement result.

## 1. Automatic detection after upload

The default live path uses Ultralytics YOLO11m-seg. It detects and segments
COCO people in every incoming frame; no first-frame box is required. Historical
service measurements below used the s profile and are labelled accordingly;
the m profile requires its own A40 benchmark before any latency claim is made.

| Quantity | Value | Meaning |
| --- | ---: | --- |
| Model parameters | 22,420,896.00 | YOLO11m-seg checkpoint (`yolo11m-seg.pt`) |
| Reference compute | 114.00 GFLOPs | Ultralytics model-info reference at 640 px |
| Model label space | 80.00 classes | COCO checkpoint classes |
| Default enabled classes | 1.00 | `person`; avoids audience and sideline-object noise |
| Configurable class mode | 80.00 | `LIVE_CLASSES=all` exposes the full checkpoint |
| Configured inference size | 960.00 px | A40 default; 640 px is the low-latency preset |
| Confidence threshold | 0.10 | Low threshold retains small/distant candidates |
| NMS maximum detections | 300.00 | Ultralytics `max_det`; not a measured scene capacity |
| Measured peak concurrent Track IDs | 23.00 | Highest count in any six-second test clip |
| Measured minimum per-frame Track IDs | 9.00 | Lowest count among tested clips; depends on the shot |
| Historical s-profile A40 rate | 15.00–15.60 FPS | Seven full six-second clips, all-lightweight path |
| Historical s-profile inference | 57.80–60.70 ms | Per-frame service time; not a YOLO11m claim |

The 23-person result is the tested scene capacity, not a promise that every
camera can detect 23 people. Small subjects, blur, extreme crowd density and
the selected class list determine recall. The model output has no hard-coded
“football player” limit.

## 2. Lightweight instance Mask, Box and persistent ID

Each YOLO detection returns a bounding box, confidence and an instance contour.
ByteTrack proposes associations, while `StableTrackRegistry` owns public,
monotonic Track IDs; SAM never changes a Track ID.

| Quantity | Value | Meaning |
| --- | ---: | --- |
| Frames with at least one lightweight Mask | 630.00 / 630.00 | Seven clips x 90 frames |
| Frames with at least one Track | 630.00 / 630.00 | Same regression run |
| Concurrent IDs per clip | 16.00–23.00 peak | Depends on the camera segment |
| Motion history capacity | 90.00 samples | Six seconds at 15 FPS per Track |
| Drawn trail length | 45.00 points | Three seconds of recent history |
| Missing-frame retention | 45.00 frames | About three seconds at 15 FPS; predicted state is explicit |
| Typical lightweight contour size | 27.00 vertices | Measured on Track 1 in the opening clip |

The lightweight contour is a compact polygon for real-time transport and
Canvas rendering. It is not the raw per-pixel tensor and should not be used as
a labelled segmentation benchmark.

## 3. Click-to-SAM refinement

Clicking a Box or Track chip changes only the selected Track to SAM 2.1 Hiera
Base+. All other Tracks remain on the lightweight path.

| Quantity | Value | Meaning |
| --- | ---: | --- |
| SAM input | 1.00 selected Box | One image prompt per incoming frame |
| Raw mask coordinate space | 1280 x 720 px | Same as the decoded source frame |
| SAM polygon vertices | 32.00–33.00 | Selected Track 1, two consecutive football frames |
| Selected-mask availability | 89.00 / 90.00 frames | Full six-second selected-SAM test |
| Mean selected-SAM latency | 145.80 ms | A40, including decode, YOLO and SAM refinement |
| Selected-SAM processing rate | 6.60 FPS | Measured service rate, not 15 FPS |
| Lightweight paths preserved | 100.00% of frames | Other visible Tracks continued returning Masks |

The selected path is a quality refinement path, not a full-frame 15 FPS SAM
claim. For a strict 15 FPS budget, keep `ALL MASKS` active and use SAM on a
selected Track intermittently or cache the last refined contour.

## 4. Tracking and movement telemetry

Live mode reports pixel speed because it has no reliable sport-specific
Homography. Offline football mode can report metric speed only when calibration
is valid.

| Quantity | Value | Meaning |
| --- | ---: | --- |
| Foot-point definition | 1.00 | Bottom-centre of the detector Box |
| Smoothing | 0.35 alpha | EMA used by `MotionHistory` |
| Pixel speed unit | px/s | Honest camera-space velocity for generic scenes |
| Metric speed without calibration | null | UI displays `Calibration required`, never fake `0 km/h` |
| Offline calibrated tracks with speed | 43.00 / 44.00 | Validated A40 full result bundle |
| Offline calibration valid frames | 450.00 / 450.00 | 100.00% in the validated bundle |
| Offline impossible short jumps | 0.00 | Validator result |
| Offline gaps over three seconds | 0.00 | Validator result |

The browser breaks a live trail when a displacement exceeds 18.00% of the
image diagonal. This prevents a bad association from drawing a false line
across the pitch; it does not silently repair the underlying Track identity.

## 5. Browser interaction and latency behavior

The browser sends at most one JPEG frame in flight. When A40 inference is busy,
the capture loop drops a newer opportunity instead of building an old-frame
queue.

| Quantity | Value |
| --- | ---: |
| Transport | Binary WebSocket |
| Frame packet header | 12.00 bytes (`uint32 frame_id` + `float64 timestamp`) |
| Lightweight A40 service rate | 15.00–15.60 FPS |
| Selected-SAM A40 service rate | 6.60 FPS |
| Browser rendering mode | Exact response frame, then Canvas overlays |
| Default display modes | ALL MASKS, SELECTED ONLY, BOXES |

The measured latency excludes camera sensor capture, browser scheduling and
SSH network round-trip. It is the inference-service latency visible to the
front end; end-user glass-to-glass latency will be higher.

## 6. Offline high-quality artifact pipeline

The validated A40 bundle (`v2-full-r3`) processed the complete 30-second source
with windowed SAM and field-space tracking.

| Quantity | Value |
| --- | ---: |
| Source frames | 450.00 |
| Retained Tracks | 44.00 |
| Mask frames | 8,296.00 |
| Mask-box IoU mean / median / p10 | 0.60 / 0.64 / 0.28 |
| Mask centroid inside Box | 91.35% |
| Mask coverage mean / median | 67.83% / 69.00% |
| Median Track lifetime | 18.80 s |
| Visible people min / median / max | 13.00 / 23.00 / 39.00 |
| End-to-end worker time | 878.67 s |
| Effective offline throughput | 0.51 FPS |
| Average / peak GPU utilization | 39.32% / 100.00% |
| Peak reported GPU memory | 15,561.00 MB |
| Active / dense object-frames | 12,194.00 / 19,800.00 |
| Object-frame reduction | 38.41% |

The IoU above compares a generated Mask with its detector Box for structural
consistency. It is **not** human-annotated segmentation IoU. No IDF1 or human
Mask IoU is claimed without a labelled validation set.

## 7. Test coverage and interpretation

- 7.00 generated media fixtures: five sequential clips and two overlapping
  boundary clips.
- 630.00 total live regression frames: every frame returned Tracks and a
  lightweight Mask.
- 90.00 selected-SAM frames: 89.00 returned a SAM Mask.
- 81.00 backend tests, 10.00 frontend tests, ESLint and production build:
  all passed.
- Controller and A40 live `/health` endpoints: passed.

The fixtures are inference/regression material, not supervised training data.
To train a domain detector or improve segmentation quality, add licensed clips
with person/object Boxes and instance-Mask annotations and evaluate recall,
precision, IDF1 and human-mask IoU separately.
