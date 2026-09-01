import argparse
import gzip
import heapq
import json
import os
import shutil
import subprocess
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .analytics import (
    bbox_touches_edge,
    classify_team,
    compute_homography,
    dominant_jersey_bgr,
    occlusion_metrics,
    ocr_vote,
    project_point,
    speed_series,
    traveled_distance,
)
from .rle import encode_mask, mask_bbox, mask_foot_point


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def monitor_gpu(stop: threading.Event, samples: list[tuple[float, float]], gpu_id: str) -> None:
    while not stop.is_set():
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--id", gpu_id,
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        utilization, memory_used = output.split(",", maxsplit=1)
        samples.append((float(utilization), float(memory_used)))
        stop.wait(0.5)


def normalize_video(source: Path, output: Path) -> None:
    run(
        [
            "ffmpeg", "-y", "-i", str(source), "-t", "60", "-vf",
            "fps=15,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", str(output),
        ]
    )


def extract_frames(video: Path, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-i", str(video), "-q:v", "2", str(directory / "%06d.jpg")])
    return sorted(directory.glob("*.jpg"))


def build_predictor():
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    checkpoint = os.environ["SAM2_CHECKPOINT"]
    model_config = os.getenv("SAM2_MODEL_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    predictor = build_sam2_video_predictor(
        model_config,
        checkpoint,
        device=device,
        hydra_overrides_extra=[
            "++model.non_overlap_masks=true",
            "++model.non_overlap_masks_for_mem_enc=true",
        ],
        vos_optimized=False,
    )
    return predictor, device


def precision_context(device: Any):
    import torch

    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def sharpness(crop: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) if crop.size else 0.0


def torso_crop(frame: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    crop_y2 = y1 + max(1, int((y2 - y1) * 0.7))
    crop = frame[y1:crop_y2, x1:x2].copy()
    crop_mask = mask[y1:crop_y2, x1:x2]
    crop[~crop_mask] = 0
    return crop


def read_numbers(crops: dict[int, list[tuple[float, np.ndarray]]]) -> dict[int, tuple[int | None, float]]:
    try:
        import easyocr
    except ImportError:
        return {object_id: (None, 0.0) for object_id in crops}
    reader = easyocr.Reader(["en"], gpu=True)
    output: dict[int, tuple[int | None, float]] = {}
    for object_id, ranked in crops.items():
        candidates: list[tuple[str, float]] = []
        for _, crop in sorted(ranked, key=lambda item: item[0], reverse=True):
            enhanced = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            for _, text, confidence in reader.readtext(enhanced, allowlist="0123456789", detail=1):
                candidates.append((text, float(confidence)))
        output[object_id] = ocr_vote(candidates)
    return output


def match_name(roster: list[dict[str, Any]], team: str, number: int | None) -> str | None:
    if number is None:
        return None
    for player in roster:
        if player["team"] == team and int(player["squad_number"]) == number:
            return player["player_name"]
    return None


def process(job_dir: Path) -> None:
    started = time.perf_counter()
    results = job_dir / "results"
    results.mkdir(exist_ok=True)
    payload = json.loads((job_dir / "payload.json").read_text())
    normalized = results / "normalized.mp4"
    frames_dir = job_dir / "frames"
    normalize_video(job_dir / "source.mp4", normalized)
    frame_paths = extract_frames(normalized, frames_dir)
    if not frame_paths:
        raise RuntimeError("video contains no frames")

    first_frame = cv2.imread(str(frame_paths[0]))
    height, width = first_frame.shape[:2]
    homography = compute_homography(payload["calibration"], width, height)
    preprocessing_seconds = time.perf_counter() - started

    model_started = time.perf_counter()
    predictor, device = build_predictor()
    import torch

    with torch.inference_mode(), precision_context(device):
        state = predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
        for prompt in payload["prompts"]:
            x1, y1, x2, y2 = prompt["box"]
            box = np.asarray([x1 * width, y1 * height, x2 * width, y2 * height], dtype=np.float32)
            center_x = (box[0] + box[2]) / 2
            box_height = box[3] - box[1]
            points = np.asarray(
                [[center_x, box[1] + box_height * 0.35], [center_x, box[1] + box_height * 0.65]],
                dtype=np.float32,
            )
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=int(prompt["object_id"]),
                box=box,
                points=points,
                labels=np.ones(2, dtype=np.int32),
            )
    model_setup_seconds = time.perf_counter() - model_started

    masks_manifest: dict[str, Any] = {"fps": 15, "width": width, "height": height, "frames": []}
    samples: dict[int, list[dict[str, Any]]] = {int(prompt["object_id"]): [] for prompt in payload["prompts"]}
    colors: dict[int, list[tuple[int, int, int]]] = {object_id: [] for object_id in samples}
    crop_heaps: dict[int, list[tuple[float, int, np.ndarray]]] = {object_id: [] for object_id in samples}
    last_seen: dict[int, tuple[int, tuple[int, int, int, int]]] = {}
    exited_ids: set[int] = set()
    writer = cv2.VideoWriter(str(results / "foreground-temp.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (width, height))
    color_index = 0

    gpu_samples: list[tuple[float, float]] = []
    monitor_stop = threading.Event()
    monitor = None
    if device.type == "cuda":
        gpu_id = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",", maxsplit=1)[0]
        monitor = threading.Thread(target=monitor_gpu, args=(monitor_stop, gpu_samples, gpu_id), daemon=True)
        monitor.start()
    propagation_started = time.perf_counter()
    with torch.inference_mode(), precision_context(device):
        propagation = iter(predictor.propagate_in_video(state))
        while True:
            try:
                frame_index, object_ids, mask_logits = next(propagation)
            except StopIteration:
                break
            frame = cv2.imread(str(frame_paths[frame_index]))
            combined = np.zeros((height, width), dtype=bool)
            frame_objects: dict[str, Any] = {}
            mask_batch = (mask_logits > 0).detach().cpu().numpy().squeeze(1).astype(bool)
            for position, object_id_value in enumerate(object_ids):
                object_id = int(object_id_value)
                if object_id in exited_ids:
                    continue
                mask = mask_batch[position]
                bbox = mask_bbox(mask)
                foot = mask_foot_point(mask)
                if bbox is None or foot is None:
                    continue
                previous = last_seen.get(object_id)
                if previous and frame_index > previous[0] + 1 and bbox_touches_edge(previous[1], width, height):
                    exited_ids.add(object_id)
                    continue
                last_seen[object_id] = (int(frame_index), bbox)
                combined |= mask
                pitch_point = project_point(foot, homography)
                mask_ys, mask_xs = np.where(mask)
                sample = {
                    "frame": int(frame_index),
                    "time": round(frame_index / 15, 3),
                    "bbox": list(bbox),
                    "foot": [round(foot[0], 2), round(foot[1], 2)],
                    "centroid": [round(float(mask_xs.mean()), 2), round(float(mask_ys.mean()), 2)],
                    "pitch": list(pitch_point),
                    "area": int(mask.sum()),
                }
                samples[object_id].append(sample)
                if frame_index % 2 == 0:
                    jersey = dominant_jersey_bgr(frame, mask, bbox)
                    if jersey:
                        colors[object_id].append(jersey)
                    crop = torso_crop(frame, mask, bbox)
                    score = sharpness(crop)
                    color_index += 1
                    entry = (score, color_index, crop)
                    if len(crop_heaps[object_id]) < 8:
                        heapq.heappush(crop_heaps[object_id], entry)
                    elif score > crop_heaps[object_id][0][0]:
                        heapq.heapreplace(crop_heaps[object_id], entry)
                frame_objects[str(object_id)] = encode_mask(mask)
            foreground = np.zeros_like(frame)
            foreground[combined] = frame[combined]
            writer.write(foreground)
            masks_manifest["frames"].append({"index": int(frame_index), "objects": frame_objects})
    propagation_seconds = time.perf_counter() - propagation_started
    monitor_stop.set()
    if monitor:
        monitor.join()
    writer.release()

    run([
        "ffmpeg", "-y", "-i", str(results / "foreground-temp.mp4"),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        str(results / "foreground.mp4"),
    ])
    (results / "foreground-temp.mp4").unlink()

    ocr_results = read_numbers({key: [(score, crop) for score, _, crop in heap] for key, heap in crop_heaps.items()})
    overlaps = occlusion_metrics(samples, frame_size=(width, height))
    references = payload["team_colors"]
    tracks: list[dict[str, Any]] = []
    for object_id, track_samples in samples.items():
        average_color = tuple(int(value) for value in np.median(colors[object_id], axis=0)) if colors[object_id] else None
        team = classify_team(average_color, references, payload["team_a"], payload["team_b"])
        number, ocr_confidence = ocr_results[object_id]
        pitch_points = [tuple(sample["pitch"]) for sample in track_samples]
        speeds = speed_series(pitch_points, 15)
        trajectory = [
            {**sample, "speed_kmh": speeds[index]} for index, sample in enumerate(track_samples)
        ]
        role = "referee" if team == "Referee" else "player"
        tracks.append(
            {
                "object_id": object_id,
                "role": role,
                "team": team,
                "jersey_number": number,
                "player_name": match_name(payload["roster"], team, number),
                "dominant_color": list(average_color) if average_color else None,
                "trajectory": trajectory,
                "speed_series": speeds,
                "metrics": {
                    **overlaps[object_id],
                    "ocr_confidence": ocr_confidence,
                    "distance_m": traveled_distance(pitch_points),
                    "average_speed_kmh": round(float(np.mean(speeds)), 2) if speeds else 0,
                    "max_speed_kmh": max(speeds, default=0),
                },
            }
        )

    with gzip.open(results / "masks.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(masks_manifest, handle, separators=(",", ":"))
    (results / "tracks.json").write_text(json.dumps(tracks, ensure_ascii=False, separators=(",", ":")))
    elapsed = time.perf_counter() - started
    gpu_peak_memory_mb = 0.0
    gpu_peak_reserved_mb = 0.0
    if str(device).startswith("cuda"):
        gpu_peak_memory_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
        gpu_peak_reserved_mb = round(torch.cuda.max_memory_reserved() / 1024**2, 2)
    gpu_utilizations = [sample[0] for sample in gpu_samples]
    gpu_memory_samples = [sample[1] for sample in gpu_samples]
    metrics = {
        "device": str(device),
        "frames": len(frame_paths),
        "objects": len(samples),
        "elapsed_seconds": round(elapsed, 2),
        "effective_fps": round(len(frame_paths) / elapsed, 2),
        "preprocessing_seconds": round(preprocessing_seconds, 2),
        "model_setup_seconds": round(model_setup_seconds, 2),
        "propagation_seconds": round(propagation_seconds, 2),
        "propagation_fps": round(len(frame_paths) / propagation_seconds, 2),
        "precision": "bfloat16" if device.type == "cuda" else "float32",
        "vos_optimized": False,
        "gpu_utilization_average_percent": round(float(np.mean(gpu_utilizations)), 2) if gpu_utilizations else 0,
        "gpu_utilization_peak_percent": max(gpu_utilizations, default=0),
        "gpu_memory_used_peak_mb": max(gpu_memory_samples, default=0),
        "gpu_peak_memory_mb": gpu_peak_memory_mb,
        "gpu_peak_reserved_mb": gpu_peak_reserved_mb,
        "occlusion_events": sum(track["metrics"]["occlusion_count"] for track in tracks),
        "ids_retained": sum(bool(track["metrics"]["id_retained"]) for track in tracks),
    }
    (results / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False))
    shutil.rmtree(frames_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", type=Path)
    process(parser.parse_args().job_dir)
