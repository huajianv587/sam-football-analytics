"""Refine one already-tracked player with the SAM 2.1 Large checkpoint."""

import argparse
import gzip
import json
import shutil
from pathlib import Path

import cv2

from .main import build_predictor, extract_frames, propagate_masks


def refine(job_dir: Path) -> None:
    payload = json.loads((job_dir / "payload.json").read_text())
    frames_dir = job_dir / "frames"
    frame_paths = extract_frames(job_dir / "normalized.mp4", frames_dir)
    track_id = int(payload["object_id"])
    detections = [
        {
            **item,
            "frame": int(item["frame"]),
            "bbox": [float(value) for value in item["bbox"]],
            "confidence": float(item.get("confidence") or 1),
            "role": payload.get("role", "player"),
        }
        for item in payload["detections"]
    ]
    predictor, device = build_predictor()
    masks = propagate_masks(predictor, device, frame_paths, {track_id: detections})[track_id]
    capture = cv2.VideoCapture(str(job_dir / "normalized.mp4"))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 15)
    capture.release()
    if not masks:
        raise RuntimeError("SAM Large refinement produced no valid masks")
    result_dir = job_dir / "results"
    result_dir.mkdir(exist_ok=True)
    manifest = {
        "track_id": track_id,
        "fps": fps,
        "width": width,
        "height": height,
        "first_frame": min(masks),
        "last_frame": max(masks),
        "model_tier": "large",
        "frames": [
            {"index": frame_index, "rle": rle}
            for frame_index, rle in sorted(masks.items())
        ],
    }
    with gzip.open(result_dir / f"{track_id}.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(manifest, handle, separators=(",", ":"))
    (result_dir / "refinement.json").write_text(
        json.dumps({"object_id": track_id, "model_tier": "large", "mask_frames": len(masks)})
    )
    shutil.rmtree(frames_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", type=Path)
    refine(parser.parse_args().job_dir)
