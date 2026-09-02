# Human validation set

`annotation-template.json` is intentionally empty of generated labels. Copy it,
sample exactly 30 frames from one continuous result clip, and label every
on-pitch person with a stable `gt_id` and pixel `bbox`.

- Add `mask_rle` for the people whose segmentation IoU will be measured.
- Add surveyed or manually controlled `pitch` coordinates only when an
  independent field reference exists.
- Never copy model boxes, Masks, Track IDs or projected positions into this
  file; doing so would turn the evaluation into a self-consistency check.

Run the evaluator from the repository root:

```bash
PYTHONPATH=backend backend/.venv/bin/python \
  backend/scripts/evaluate_ground_truth.py \
  /path/to/results /path/to/human-annotations.json \
  --output /path/to/ground-truth-report.json
```

The report contains detection precision/recall, global IDF1, human-mask mean
IoU and median pitch-projection error for the evidence actually supplied.
