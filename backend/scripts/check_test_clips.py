#!/usr/bin/env python3
"""Deep-check generated clips without running a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def check(path: Path, expected_frames: int, expected_fps: float) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    decoded = 0
    sampled_nonempty = 0
    previous_pts = -1.0
    pts_regressions = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded += 1
        if decoded == 1 or decoded % max(1, round(expected_fps)) == 0:
            if frame.size and float(frame.mean()) > 0.0:
                sampled_nonempty += 1
        pts = capture.get(cv2.CAP_PROP_POS_MSEC)
        if pts + 0.001 < previous_pts:
            pts_regressions += 1
        previous_pts = pts
    capture.release()
    result = {
        "path": str(path),
        "decoded_frames": decoded,
        "expected_frames": expected_frames,
        "sampled_nonempty_frames": sampled_nonempty,
        "timestamp_regressions": pts_regressions,
        "pass": decoded == expected_frames and sampled_nonempty > 0 and pts_regressions == 0,
    }
    if not result["pass"]:
        raise RuntimeError(json.dumps(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checks = []
    for item in manifest["clips"]:
        checks.append(check(Path(item["path"]), item["frames"], 15.0))
    report = {"schema": "pitchvision.media-check.v1", "clips": checks, "pass": all(item["pass"] for item in checks)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
