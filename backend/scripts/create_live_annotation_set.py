#!/usr/bin/env python3
"""Create an independently labelable 30-frame package from a live session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path, help="live-session cache directory")
    parser.add_argument("output", type=Path, help="empty destination directory")
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()
    if args.frames < 2:
        raise SystemExit("--frames must be at least 2")
    source = args.session / "source.mp4"
    if not source.is_file():
        raise SystemExit(f"source video was not found: {source}")
    capture = cv2.VideoCapture(str(source))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not total or not width or not height:
        raise SystemExit("source video has no readable frames")
    selected = sorted({round(index * (total - 1) / (args.frames - 1)) for index in range(args.frames)})
    frames_dir = args.output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    annotations = []
    for frame_index in selected:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise SystemExit(f"unable to decode frame {frame_index}")
        image_name = f"frame-{frame_index:04d}.jpg"
        if not cv2.imwrite(str(frames_dir / image_name), frame):
            raise SystemExit(f"unable to write frame {frame_index}")
        annotations.append({"frame": frame_index, "image": f"frames/{image_name}", "objects": []})
    capture.release()
    (args.output / "annotations.json").write_text(json.dumps({
        "description": "Independently label every visible person. Do not copy model boxes, Masks or IDs.",
        "width": width,
        "height": height,
        "frames": annotations,
    }, indent=2) + "\n")
    print(f"wrote {len(annotations)} frames to {args.output}")


if __name__ == "__main__":
    main()
