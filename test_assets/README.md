# Local video evaluation fixtures

Run the following from the repository root to create independent MP4 clips
from the supplied long shot:

```bash
python3 backend/scripts/split_test_clips.py \
  "视频素材/标准化/西班牙_阿根廷_连续镜头_30s_720p15.mp4"
```

Outputs are written to `test_assets/generated/` (ignored by Git) and described
by `manifest.json`. The five sequential six-second clips cover the full source;
the two additional six-second clips overlap adjacent segments to exercise
temporal stitching and window-boundary behavior. Every clip is re-encoded as
H.264/yuv420p with fast-start metadata so it can be opened independently by a
browser, OpenCV, and FFmpeg.

These files are **not a supervised training set**: the source has no frame
labels. They are safe, reproducible inference and regression fixtures. For
training, add licensed footage together with person/instance masks and a
versioned annotation manifest.

The deep media check validates codec, dimensions, frame count, duration,
monotonic decoding, and non-empty frames for every generated clip.

Run it with:

```bash
python3 backend/scripts/check_test_clips.py test_assets/generated/manifest.json
```
