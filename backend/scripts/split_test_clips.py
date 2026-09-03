#!/usr/bin/env python3
"""Create reproducible, independently decodable clips for local evaluation.

The source video is never modified. Generated clips are intentionally kept out
of git because the footage may be copyrighted and contains no ground-truth
labels; they are evaluation/inference fixtures, not a supervised training set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


SEGMENTS = (
    ("clip_01_opening", 0.0, 6.0),
    ("clip_02_build_up", 6.0, 6.0),
    ("clip_03_midfield", 12.0, 6.0),
    ("clip_04_transition", 18.0, 6.0),
    ("clip_05_attack", 24.0, 6.0),
    # One-second overlap makes this useful for testing temporal stitching.
    ("eval_overlap_05_11", 5.0, 6.0),
    ("eval_overlap_17_23", 17.0, 6.0),
)


def run(*args: str) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout


def probe(path: Path) -> dict:
    raw = run(
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    )
    data = json.loads(raw)
    stream = next((item for item in data.get("streams", []) if item.get("codec_name")), {})
    fmt = data.get("format", {})
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": stream.get("avg_frame_rate") or stream.get("r_frame_rate"),
        "frames": int(stream["nb_frames"]) if stream.get("nb_frames") else None,
        "duration_s": round(float(fmt.get("duration") or stream.get("duration") or 0), 3),
        "bytes": int(fmt.get("size") or path.stat().st_size),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("test_assets/generated"))
    args = parser.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise SystemExit("ffmpeg and ffprobe are required")
    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"source does not exist: {source}")
    args.output.mkdir(parents=True, exist_ok=True)

    source_meta = probe(source)
    clips: list[dict] = []
    for name, start, duration in SEGMENTS:
        destination = args.output / f"{name}.mp4"
        subprocess.run(
            (
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(source),
                "-t",
                f"{duration:.3f}",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(destination),
            ),
            check=True,
        )
        metadata = probe(destination)
        if metadata["codec"] != "h264" or metadata["width"] != 1280 or metadata["height"] != 720:
            raise SystemExit(f"invalid clip contract: {destination}")
        if metadata["frames"] is None or metadata["frames"] < 80:
            raise SystemExit(f"clip has too few frames: {destination}")
        clips.append(
            {
                "name": name,
                "start_s": start,
                "requested_duration_s": duration,
                "path": str(destination),
                "sha256": sha256(destination),
                **metadata,
            }
        )

    manifest = {
        "schema": "pitchvision.test-clips.v1",
        "purpose": "inference and evaluation fixtures; no ground-truth labels",
        "source": {"path": str(source), "sha256": sha256(source), **source_meta},
        "clips": clips,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
