import argparse
import gzip
import heapq
import json
import math
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .analytics import (
    classify_team,
    dominant_jersey_bgr,
    occlusion_metrics,
    ocr_vote,
    project_point,
    smooth_metric_positions,
    speed_series,
    traveled_distance,
)
from .game_state import (
    bbox_iou,
    detections_by_track,
    frame_calibrations,
    load_game_state,
    manual_game_state,
    on_pitch_tracks,
    select_prompt_detections,
    track_continuity_metrics,
    track_windows,
)
from .field_tracker import field_space_track, interpolate_calibrations, resample_tracks
from .rle import decode_mask, encode_mask, mask_bbox, mask_foot_point


FPS = 15


def run(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, cwd=cwd, env=env)


def set_progress(job_dir: Path, stage: str, progress: int, track_count: int | None = None) -> None:
    temporary = job_dir / "progress.json.tmp"
    payload: dict[str, Any] = {"stage": stage, "progress": progress}
    if track_count is not None:
        payload["track_count"] = track_count
    temporary.write_text(json.dumps(payload))
    temporary.replace(job_dir / "progress.json")


def monitor_gpu(stop: threading.Event, samples: list[tuple[float, float]], gpu_id: str) -> None:
    while not stop.is_set():
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--id",
                gpu_id,
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        utilization, memory_used = output.split(",", maxsplit=1)
        samples.append((float(utilization), float(memory_used)))
        stop.wait(0.5)


def normalized_dimensions(source: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(source))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if not width or not height:
        raise RuntimeError("unable to read source video dimensions")
    scale = min(1.0, 1920 / width, 1080 / height)
    return max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)


def normalize_video(source: Path, output: Path) -> None:
    width, height = normalized_dimensions(source)
    run(
        [
            "ffmpeg", "-y", "-i", str(source), "-t", "60", "-vf",
            f"fps={FPS},scale={width}:{height}", "-an", "-c:v", "libx264",
            "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", str(output),
        ]
    )


def extract_frames(video: Path, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-i", str(video), "-q:v", "2", str(directory / "%06d.jpg")])
    return sorted(directory.glob("*.jpg"))


def run_game_state_reconstruction(job_dir: Path, video: Path) -> tuple[Path, dict[str, float]]:
    runtime = Path(os.environ["GSR_RUNTIME_DIR"])
    capture = cv2.VideoCapture(str(video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    detector_fps = float(os.getenv("DETECTOR_TRACKER_FPS", "10"))
    calibration_fps = float(os.getenv("CALIBRATION_FPS", "5"))
    timings: dict[str, float] = {}

    def prepare_rate_video(name: str, analysis_fps: float) -> Path:
        if abs(analysis_fps - FPS) < 1e-6:
            return video
        output = job_dir / f"{name}-input.mp4"
        prepared = time.perf_counter()
        run(
            [
                "ffmpeg", "-y", "-i", str(video), "-vf", f"fps={analysis_fps}",
                "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-pix_fmt", "yuv420p", str(output),
            ]
        )
        timings[f"{name}_prepare_seconds"] = time.perf_counter() - prepared
        return output

    detector_video = prepare_rate_video("detector", detector_fps)
    calibration_video = prepare_rate_video("calibration", calibration_fps)
    config_dir = job_dir / "gsr"

    def run_stage(
        name: str,
        progress: int,
        pipeline: list[str],
        source_video: Path,
        save_state: Path,
    ) -> Path:
        set_progress(job_dir, name, progress)
        environment = os.environ.copy()
        environment.update(
            {
                "GSR_PROJECT_DIR": str(runtime),
                "PITCHVISION_VIDEO": str(source_video),
                "PITCHVISION_WIDTH": str(width),
                "PITCHVISION_HEIGHT": str(height),
                "PITCHVISION_STATE": str(save_state),
                "PITCHVISION_RUN_DIR": str(job_dir / f"gsr-{name}"),
            }
        )
        started = time.perf_counter()
        run(
            [
                "conda", "run", "--no-capture-output", "-p", str(runtime / "env"),
                "python", "-m", "tracklab.main", "--config-path", str(config_dir),
                "--config-name", "pitchvision",
                f"pipeline=[{','.join(pipeline)}]",
                "state.load_file=null",
                f"state.save_file={save_state}",
            ],
            cwd=runtime / "SoccerMaster" / "codes" / "sn-gamestate",
            env=environment,
        )
        timings[f"{name}_seconds"] = time.perf_counter() - started
        exported = job_dir / f"{name}-state.json"
        run(
            [
                "conda", "run", "--no-capture-output", "-p", str(runtime / "env"),
                "python", str(config_dir / "export_state.py"), str(save_state),
                str(source_video), str(exported),
            ],
            env=environment,
        )
        return exported

    detection_json = run_stage(
        "detect", 12, ["bbox_detector"], detector_video, job_dir / "detections.pklz"
    )
    calibration_json = run_stage(
        # PnLCalib's camera solver and bbox-to-pitch projector are one TrackLab
        # module, so it requires bbox_ltwh columns. A 5 FPS detector supplies
        # only that schema; ReID and association still run solely at their
        # independent rates.
        "calibrate", 28, ["bbox_detector", "pitch", "calibration"], calibration_video,
        job_dir / "calibration.pklz",
    )
    detection_state = json.loads(detection_json.read_text())
    calibration_state = json.loads(calibration_json.read_text())
    detector_calibrations = interpolate_calibrations(
        calibration_state, len(detection_state["frames"]), detector_fps
    )
    set_progress(job_dir, "track", 40)
    tracking_started = time.perf_counter()
    tracked_state, association_metrics = field_space_track(
        detection_state, detector_calibrations, detector_video
    )
    timings["field_track_seconds"] = time.perf_counter() - tracking_started
    timings.update({
        f"field_tracker_{key}": value
        for key, value in association_metrics.items()
        if isinstance(value, (int, float))
    })
    full_calibrations = interpolate_calibrations(calibration_state, frame_count, FPS)
    state = resample_tracks(tracked_state, full_calibrations, frame_count, FPS)
    state["association_metrics"] = association_metrics
    output = job_dir / "game-state.json"
    output.write_text(json.dumps(state, separators=(",", ":")))
    return output, timings


def classify_track_roles(
    job_dir: Path,
    video: Path,
    game_state_path: Path,
    track_ids: list[int],
) -> dict[int, dict[str, Any]]:
    from gsr.role_logic import resolve_role_with_pitch

    runtime = Path(os.environ["GSR_RUNTIME_DIR"])
    output = job_dir / "roles.json"
    run(
        [
            "conda", "run", "--no-capture-output", "-p", str(runtime / "env"),
            "python", str(job_dir / "gsr" / "classify_roles.py"), str(video),
            str(game_state_path), str(runtime / "pretrained_models" / "jn" / "Qwen2.5-VL-7B-Instruct"),
            str(output), ",".join(str(track_id) for track_id in track_ids),
        ]
    )
    predictions = {
        int(track_id): value for track_id, value in json.loads(output.read_text()).items()
    }
    state = load_game_state(game_state_path)
    by_track = detections_by_track(state)
    for track_id, prediction in predictions.items():
        prediction["role"] = resolve_role_with_pitch(
            prediction["role"], by_track.get(track_id, [])
        )
    return predictions


def build_predictor():
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    checkpoint = os.environ["SAM2_CHECKPOINT"]
    model_config = os.getenv("SAM2_MODEL_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    vos_optimized = os.getenv("SAM_VOS_OPTIMIZED", "true").lower() == "true"
    original_compile = torch.compile
    if vos_optimized and device.type == "cuda":
        torch.compile = sam_compile_without_cudagraphs(original_compile)
    try:
        predictor = build_sam2_video_predictor(
            model_config,
            checkpoint,
            device=device,
            hydra_overrides_extra=[
                "++model.non_overlap_masks=true",
                "++model.non_overlap_masks_for_mem_enc=true",
            ],
            vos_optimized=vos_optimized,
        )
    finally:
        torch.compile = original_compile
    return predictor, device


def sam_compile_without_cudagraphs(compile_fn: Callable[..., Any]) -> Callable[..., Any]:
    def compile_component(model: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("mode") == "max-autotune":
            kwargs = {**kwargs, "mode": "max-autotune-no-cudagraphs"}
        return compile_fn(model, *args, **kwargs)

    return compile_component


def precision_context(device: Any):
    import torch

    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def sharpness(crop: np.ndarray) -> float:
    if not crop.size:
        return 0.0
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def torso_crop(frame: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    crop_y2 = y1 + max(1, int((y2 - y1) * 0.7))
    crop = frame[y1:crop_y2, x1:x2].copy()
    crop_mask = mask[y1:crop_y2, x1:x2]
    crop[~crop_mask] = 0
    return crop


def recognize_jersey_numbers(
    crop_heaps: dict[int, list[tuple[float, int, np.ndarray]]],
    use_gpu: bool,
) -> dict[int, tuple[int | None, float]]:
    import easyocr

    reader = easyocr.Reader(["en"], gpu=use_gpu)
    track_ids = sorted(crop_heaps)
    collages = [
        jersey_collage(
            [
                item[2]
                for item in sorted(
                    crop_heaps[track_id],
                    key=lambda item: (item[0], item[1]),
                    reverse=True,
                )
            ]
        )
        for track_id in track_ids
    ]
    batches = reader.readtext_batched(
        collages,
        n_width=640,
        n_height=192,
        batch_size=32,
        allowlist="0123456789",
        detail=1,
        paragraph=False,
        min_size=8,
    )
    return {
        track_id: ocr_vote(
            [(text, float(confidence)) for _, text, confidence in detections]
        )
        for track_id, detections in zip(track_ids, batches, strict=True)
    }


def jersey_collage(
    crops: list[np.ndarray],
    tile_width: int = 128,
    tile_height: int = 192,
    limit: int = 5,
) -> np.ndarray:
    """Combine one track's clearest torso crops into a single OCR image."""
    canvas = np.zeros((tile_height, tile_width * limit, 3), dtype=np.uint8)
    for index, crop in enumerate(crops[:limit]):
        height, width = crop.shape[:2]
        scale = min((tile_width - 8) / width, (tile_height - 8) / height)
        resized = cv2.resize(
            crop,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
        x = index * tile_width + (tile_width - resized.shape[1]) // 2
        y = (tile_height - resized.shape[0]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def detection_at(detections: list[dict[str, Any]], frame_index: int) -> dict[str, Any] | None:
    return next((item for item in detections if item["frame"] == frame_index), None)


def mask_quality(mask: np.ndarray, detection: dict[str, Any] | None) -> float:
    bbox = mask_bbox(mask)
    if bbox is None:
        return -1
    return bbox_iou(bbox, detection["bbox"]) if detection else 0.5


def motion_sample(
    frame_index: int,
    detection: dict[str, Any],
    calibration: dict[str, Any],
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a metric observation even when a SAM frame is unavailable.

    Detection and calibration own the continuous trajectory. A sufficiently
    trustworthy SAM foot point can only make a small correction, so a rejected
    or missing Mask never blanks an otherwise valid speed observation.
    """
    display_bbox = [round(float(value), 2) for value in detection["bbox"]]
    detection_foot = (
        (float(detection["bbox"][0]) + float(detection["bbox"][2])) / 2,
        float(detection["bbox"][3]),
    )
    mask_foot = mask_foot_point(mask) if mask is not None else None
    quality = mask_quality(mask, detection) if mask is not None else 0.0
    bbox_height = max(1.0, float(display_bbox[3]) - float(display_bbox[1]))
    mask_correction_valid = bool(
        mask_foot is not None
        and quality >= 0.65
        and math.dist(mask_foot, detection_foot) <= 0.25 * bbox_height
    )
    foot = (
        (
            0.8 * detection_foot[0] + 0.2 * mask_foot[0],
            0.8 * detection_foot[1] + 0.2 * mask_foot[1],
        )
        if mask_correction_valid and mask_foot is not None
        else detection_foot
    )
    pitch_point = None
    if calibration.get("valid"):
        homography = np.asarray(calibration["homography"], dtype=np.float64).reshape(3, 3)
        candidate = project_point(foot, homography)
        if -1 <= candidate[0] <= 106 and -1 <= candidate[1] <= 69:
            pitch_point = candidate

    if mask is not None and mask.any():
        mask_ys, mask_xs = np.where(mask)
        centroid = [round(float(mask_xs.mean()), 2), round(float(mask_ys.mean()), 2)]
        area = int(mask.sum())
    else:
        centroid = [
            round((display_bbox[0] + display_bbox[2]) / 2, 2),
            round((display_bbox[1] + display_bbox[3]) / 2, 2),
        ]
        area = 0

    return {
        "frame": frame_index,
        "time": round(frame_index / FPS, 3),
        "bbox": display_bbox,
        "foot": [round(foot[0], 2), round(foot[1], 2)],
        "centroid": centroid,
        "pitch": list(pitch_point) if pitch_point else None,
        "area": area,
        "mask_available": mask is not None,
        "mask_quality": round(float(quality), 3),
        "position_source": "bbox+mask" if mask_correction_valid else "bbox",
    }


def propagation_directions(
    prompts: dict[int, list[dict[str, Any]]], bidirectional: bool
) -> list[tuple[int, bool]]:
    first = min(prompt["frame"] for items in prompts.values() for prompt in items)
    if not bidirectional:
        return [(first, False)]
    real_prompt_sets = [
        items for track_id, items in prompts.items() if track_id >= 0 and items
    ]
    # Start only after every real object has at least one conditioning prompt.
    # Forward and reverse then partition the window around one shared anchor;
    # starting at the earliest and latest prompts would recompute most frames.
    anchor = max(min(prompt["frame"] for prompt in items) for items in real_prompt_sets)
    return [(anchor, False), (anchor, True)]


def propagate_masks(
    predictor: Any,
    device: Any,
    frame_paths: list[Path],
    track_detections: dict[int, list[dict[str, Any]]],
    progress_callback: Callable[[float], None] | None = None,
) -> dict[int, dict[int, dict[str, Any]]]:
    import torch

    object_batch = max(1, int(os.getenv("SAM_OBJECT_BATCH", "8")))
    min_detection_iou = max(0.0, float(os.getenv("SAM_MIN_DETECTION_IOU", "0.1")))
    window_size = max(30, int(os.getenv("SAM_WINDOW_FRAMES", "180")))
    window_overlap = min(window_size - 1, max(0, int(os.getenv("SAM_WINDOW_OVERLAP", "30"))))
    bidirectional = os.getenv("SAM_BIDIRECTIONAL", "true").lower() == "true"
    prompt_count = max(1, int(os.getenv("SAM_PROMPTS_PER_TRACK", "3")))
    pad_compiled_batch = (
        os.getenv("SAM_PAD_COMPILED_BATCH", "true").lower() == "true"
        and os.getenv("SAM_VOS_OPTIMIZED", "true").lower() == "true"
        and device.type == "cuda"
    )
    masks: dict[int, dict[int, dict[str, Any]]] = {track_id: {} for track_id in track_detections}
    quality: dict[int, dict[int, float]] = {track_id: {} for track_id in track_detections}
    first = min(detections[0]["frame"] for detections in track_detections.values())
    last = max(detections[-1]["frame"] for detections in track_detections.values())
    windows = track_windows(first, last, window_size, window_overlap)
    total_passes = 0
    for window_start, window_end in windows:
        active_count = sum(
            any(window_start <= item["frame"] <= window_end for item in detections)
            for detections in track_detections.values()
        )
        passes_per_batch = 2 if bidirectional else 1
        total_passes += passes_per_batch * ((active_count + object_batch - 1) // object_batch)
    completed_passes = 0
    window_root = frame_paths[0].parent.parent / "sam-window-frames"
    shutil.rmtree(window_root, ignore_errors=True)
    window_root.mkdir()
    for window_start, window_end in windows:
        window_dir = window_root / f"{window_start:06d}-{window_end:06d}"
        window_dir.mkdir()
        for local_index, frame_path in enumerate(
            frame_paths[window_start : window_end + 1]
        ):
            (window_dir / f"{local_index:06d}.jpg").symlink_to(frame_path.resolve())
        active_ids = [
            track_id
            for track_id, detections in sorted(
                track_detections.items(),
                key=lambda item: (item[1][0]["frame"] + item[1][-1]["frame"]) / 2,
            )
            if any(window_start <= item["frame"] <= window_end for item in detections)
        ]
        for offset in range(0, len(active_ids), object_batch):
            batch_ids = active_ids[offset : offset + object_batch]
            with torch.inference_mode(), precision_context(device):
                state = predictor.init_state(
                    video_path=str(window_dir),
                    offload_video_to_cpu=False,
                    offload_state_to_cpu=False,
                    async_loading_frames=False,
                )
                prompts: dict[int, list[dict[str, Any]]] = {}
                for track_id in batch_ids:
                    candidates = [
                        item
                        for item in track_detections[track_id]
                        if window_start <= item["frame"] <= window_end
                    ]
                    if bidirectional:
                        prompts[track_id] = select_prompt_detections(
                            candidates, frame_paths, count=prompt_count
                        )
                    else:
                        first_detection = candidates[0]
                        extra_prompts = (
                            select_prompt_detections(
                                candidates, frame_paths, count=prompt_count - 1
                            )
                            if prompt_count > 1
                            else []
                        )
                        prompts[track_id] = [first_detection] + [
                            prompt
                            for prompt in extra_prompts
                            if prompt["frame"] != first_detection["frame"]
                        ]
                for track_id, track_prompts in prompts.items():
                    for prompt in track_prompts:
                        box = np.asarray(prompt["bbox"], dtype=np.float32)
                        center_x = (box[0] + box[2]) / 2
                        box_height = box[3] - box[1]
                        points = np.asarray(
                            [
                                [center_x, box[1] + box_height * 0.35],
                                [center_x, box[1] + box_height * 0.65],
                            ],
                            dtype=np.float32,
                        )
                        predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=int(prompt["frame"]) - window_start,
                            obj_id=track_id,
                            box=box,
                            points=points,
                            labels=np.ones(2, dtype=np.int32),
                        )

                # Full VOS compile is shape-sensitive. Pad only the final
                # object bucket so every compiled propagation sees exactly
                # SAM_OBJECT_BATCH identities; dummy corner masks are never
                # exported and do not overlap on-pitch people.
                if pad_compiled_batch and len(batch_ids) < object_batch:
                    for dummy_offset in range(object_batch - len(batch_ids)):
                        dummy_id = -1 - offset - dummy_offset
                        prompts[dummy_id] = [{"frame": window_start}]
                        predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=0,
                            obj_id=dummy_id,
                            box=np.asarray([0, 0, 2, 2], dtype=np.float32),
                            points=np.asarray([[1, 1]], dtype=np.float32),
                            labels=np.ones(1, dtype=np.int32),
                        )

                directions = [
                    (frame_index - window_start, reverse)
                    for frame_index, reverse in propagation_directions(
                        prompts, bidirectional
                    )
                ]
                batch_start_global = min(
                    max(window_start, track_detections[track_id][0]["frame"])
                    for track_id in batch_ids
                )
                batch_end_global = max(
                    min(window_end, track_detections[track_id][-1]["frame"])
                    for track_id in batch_ids
                )
                batch_start = batch_start_global - window_start
                batch_end = batch_end_global - window_start
                for start_frame, reverse in directions:
                    max_frames = (
                        batch_end - start_frame + 1
                        if not reverse
                        else start_frame - batch_start + 1
                    )
                    propagation = predictor.propagate_in_video(
                        state,
                        start_frame_idx=int(start_frame),
                        max_frame_num_to_track=max_frames,
                        reverse=reverse,
                    )
                    for pass_step, (local_frame_index, object_ids, mask_logits) in enumerate(
                        propagation, start=1
                    ):
                        frame_index = int(local_frame_index) + window_start
                        if progress_callback:
                            progress_callback(
                                (completed_passes + min(1.0, pass_step / max_frames))
                                / max(1, total_passes)
                            )
                        if not window_start <= frame_index <= window_end:
                            continue
                        active = [
                            (position, int(object_id_value))
                            for position, object_id_value in enumerate(object_ids)
                            if int(object_id_value) in track_detections
                            and track_detections[int(object_id_value)][0]["frame"]
                            <= frame_index
                            <= track_detections[int(object_id_value)][-1]["frame"]
                        ]
                        if not active:
                            continue
                        active_positions = torch.as_tensor(
                            [position for position, _ in active],
                            device=mask_logits.device,
                        )
                        mask_batch = (
                            (mask_logits.index_select(0, active_positions) > 0)
                            .detach()
                            .cpu()
                            .numpy()
                            .squeeze(1)
                            .astype(bool)
                        )
                        for mask_position, (_, track_id) in enumerate(active):
                            detections = track_detections[track_id]
                            mask = mask_batch[mask_position]
                            detection = detection_at(detections, int(frame_index))
                            score = mask_quality(mask, detection)
                            if detection and score < min_detection_iou:
                                continue
                            if score > quality[track_id].get(int(frame_index), -1):
                                masks[track_id][int(frame_index)] = encode_mask(mask)
                                quality[track_id][int(frame_index)] = score
                    completed_passes += 1
                del state
        shutil.rmtree(window_dir)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    shutil.rmtree(window_root)
    return masks


def aggregate_detection_metadata(detections: list[dict[str, Any]]) -> dict[str, Any]:
    role_scores: dict[str, float] = {}
    jersey_scores: dict[int, float] = {}
    for detection in detections:
        role = detection.get("role") or "player"
        role_scores[role] = role_scores.get(role, 0) + float(
            detection.get("role_confidence") or detection.get("confidence") or 0
        )
        number = detection.get("jersey_number")
        if number is not None:
            jersey_scores[int(number)] = jersey_scores.get(int(number), 0) + float(
                detection.get("identity_confidence") or 0
            )
    role = max(role_scores, key=role_scores.get) if role_scores else "player"
    number = max(jersey_scores, key=jersey_scores.get) if jersey_scores else None
    confidence = min(1.0, jersey_scores.get(number, 0)) if number is not None else 0.0
    return {"role": role, "jersey_number": number, "identity_confidence": confidence}


def roster_match(
    roster: list[dict[str, Any]], team: str, number: int | None, confidence: float
) -> dict[str, Any] | None:
    if number is None or confidence < 0.65:
        return None
    return next(
        (
            player
            for player in roster
            if player["team"] == team and int(player["squad_number"]) == number
        ),
        None,
    )


def has_configured_fixture(payload: dict[str, Any]) -> bool:
    """Return false for the direct-upload generic demo context."""
    return not (
        str(payload.get("match_label") or "").strip().lower() == "unspecified match"
        and str(payload.get("team_a") or "").strip().lower() == "team a"
        and str(payload.get("team_b") or "").strip().lower() == "team b"
    )


def merge_jersey_predictions(
    first: tuple[int | None, float], second: tuple[int | None, float]
) -> tuple[int | None, float]:
    """Fuse independent OCR predictions without guessing across disagreements."""
    first_number, first_confidence = first
    second_number, second_confidence = second
    if first_number is None:
        return second
    if second_number is None:
        return first
    if first_number == second_number:
        confidence = 1 - (1 - first_confidence) * (1 - second_confidence)
        return first_number, round(min(1.0, confidence), 3)
    return first if first_confidence >= second_confidence else second


def fast_role_from_appearance(
    provisional_team: str, detections: list[dict[str, Any]]
) -> tuple[str, float]:
    if provisional_team == "Referee":
        return "referee", 0.9
    if provisional_team != "unknown":
        return "player", 0.9
    pitch_x = [
        float(item["pitch"]["x_bottom_middle"])
        for item in detections
        if isinstance(item.get("pitch"), dict)
        and item["pitch"].get("x_bottom_middle") is not None
    ]
    if pitch_x and abs(float(np.median(pitch_x))) >= 40:
        return "goalkeeper", 0.65
    return "player", 0.5


def process(job_dir: Path) -> None:
    started = time.perf_counter()
    stage_times: dict[str, float] = {}
    results = job_dir / "results"
    results.mkdir(exist_ok=True)
    payload = json.loads((job_dir / "payload.json").read_text())
    normalized = results / "normalized.mp4"
    frames_dir = job_dir / "frames"
    gpu_samples: list[tuple[float, float]] = []
    monitor_stop = threading.Event()
    monitor = None
    if os.getenv("CUDA_VISIBLE_DEVICES"):
        gpu_id = os.environ["CUDA_VISIBLE_DEVICES"].split(",", maxsplit=1)[0]
        monitor = threading.Thread(
            target=monitor_gpu, args=(monitor_stop, gpu_samples, gpu_id), daemon=True
        )
        monitor.start()

    set_progress(job_dir, "normalize", 5)
    stage_started = time.perf_counter()
    normalize_video(job_dir / "source.mp4", normalized)
    frame_paths = extract_frames(normalized, frames_dir)
    if not frame_paths:
        raise RuntimeError("video contains no frames")
    first_frame = cv2.imread(str(frame_paths[0]))
    height, width = first_frame.shape[:2]
    stage_times["normalize_seconds"] = time.perf_counter() - stage_started

    if payload.get("analysis_mode") == "auto_all":
        stage_started = time.perf_counter()
        game_state_path, game_state_times = run_game_state_reconstruction(job_dir, normalized)
        stage_times["game_state_seconds"] = time.perf_counter() - stage_started
        stage_times.update(game_state_times)
        game_state = load_game_state(game_state_path)
    else:
        game_state = manual_game_state(payload, len(frame_paths), width, height)

    track_detections = detections_by_track(game_state)
    raw_track_count = len(track_detections)
    if payload.get("analysis_mode") == "auto_all":
        track_detections = on_pitch_tracks(track_detections)
    if not track_detections:
        raise RuntimeError("no on-pitch person tracks remain after filtering")
    tracking_metrics = track_continuity_metrics(track_detections)
    set_progress(job_dir, "calibrate", 44, len(track_detections))
    calibrations = frame_calibrations(game_state)
    calibration_payload = {
        "fps": FPS,
        "width": width,
        "height": height,
        "frames": [
            {
                "index": index,
                **calibrations.get(index, {"homography": None, "confidence": 0, "valid": False}),
            }
            for index in range(len(frame_paths))
        ],
    }
    if payload.get("analysis_mode") == "auto_all":
        with gzip.open(results / "calibration.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(calibration_payload, handle, separators=(",", ":"))

    set_progress(job_dir, "segment", 50, len(track_detections))
    stage_started = time.perf_counter()
    predictor, device = build_predictor()
    reported_progress = 50

    def report_segment_progress(fraction: float) -> None:
        nonlocal reported_progress
        progress = min(80, 50 + int(fraction * 30))
        if progress > reported_progress:
            reported_progress = progress
            set_progress(job_dir, "segment", progress, len(track_detections))

    encoded_masks = propagate_masks(
        predictor,
        device,
        frame_paths,
        track_detections,
        progress_callback=report_segment_progress,
    )
    stage_times["segment_seconds"] = time.perf_counter() - stage_started
    maskless_track_ids = sorted(
        track_id for track_id, track_masks in encoded_masks.items() if not track_masks
    )
    if maskless_track_ids:
        track_detections = {
            track_id: detections
            for track_id, detections in track_detections.items()
            if track_id not in maskless_track_ids
        }
        encoded_masks = {
            track_id: track_masks
            for track_id, track_masks in encoded_masks.items()
            if track_id not in maskless_track_ids
        }
    if not track_detections:
        raise RuntimeError("SAM produced no usable on-pitch person masks")
    del predictor
    if device.type == "cuda":
        import torch

        torch.cuda.empty_cache()

    set_progress(job_dir, "identify", 82, len(track_detections))
    mask_postprocess_started = time.perf_counter()
    samples: dict[int, list[dict[str, Any]]] = {track_id: [] for track_id in track_detections}
    colors: dict[int, list[tuple[int, int, int]]] = {track_id: [] for track_id in track_detections}
    crop_heaps: dict[int, list[tuple[float, int, np.ndarray]]] = {
        track_id: [] for track_id in track_detections
    }
    crop_index = 0
    detections_per_frame: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for track_id, detections in track_detections.items():
        for detection in detections:
            detections_per_frame.setdefault(int(detection["frame"]), []).append(
                (track_id, detection)
            )
    for frame_index, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        calibration = calibrations.get(frame_index, {})
        for track_id, detection in detections_per_frame.get(frame_index, []):
            rle = encoded_masks[track_id].get(frame_index)
            mask = decode_mask(rle).astype(bool) if rle else None
            sample = motion_sample(frame_index, detection, calibration, mask)
            samples[track_id].append(sample)
            bbox = mask_bbox(mask) if mask is not None else None
            if frame_index % 2 == 0 and mask is not None and bbox is not None:
                jersey = dominant_jersey_bgr(frame, mask, bbox)
                if jersey:
                    colors[track_id].append(jersey)
                crop = torso_crop(frame, mask, bbox)
                crop_index += 1
                entry = (sharpness(crop), crop_index, crop)
                if len(crop_heaps[track_id]) < 8:
                    heapq.heappush(crop_heaps[track_id], entry)
                elif entry[0] > crop_heaps[track_id][0][0]:
                    heapq.heapreplace(crop_heaps[track_id], entry)
    stage_times["mask_postprocess_seconds"] = time.perf_counter() - mask_postprocess_started

    identity_started = time.perf_counter()
    configured_fixture = has_configured_fixture(payload)
    if payload.get("analysis_mode") == "auto_all":
        role_candidates = []
        provisional_teams: dict[int, str] = {}
        for track_id, track_colors in colors.items():
            average = (
                tuple(int(value) for value in np.median(track_colors, axis=0))
                if track_colors
                else None
            )
            provisional_team = (
                classify_team(
                    average, payload["team_colors"], payload["team_a"], payload["team_b"]
                )
                if configured_fixture
                else "unknown"
            )
            provisional_teams[track_id] = provisional_team
            if (
                os.getenv("ROLE_MODEL_ENABLED", "false").lower() == "true"
                and provisional_team in {"unknown", "Referee"}
            ):
                role_candidates.append(track_id)
        role_started = time.perf_counter()
        role_predictions = (
            classify_track_roles(job_dir, normalized, game_state_path, role_candidates)
            if role_candidates
            else {}
        )
        stage_times["role_seconds"] = time.perf_counter() - role_started
        for track_id, detections in track_detections.items():
            prediction = role_predictions.get(track_id)
            role, confidence = (
                (prediction["role"], prediction["confidence"])
                if prediction
                else fast_role_from_appearance(
                    provisional_teams.get(track_id, "unknown"), detections
                )
            )
            for detection in detections:
                detection["role"] = role
                detection["role_confidence"] = confidence

    foreground_ids = {
        track_id
        for track_id, detections in track_detections.items()
        if aggregate_detection_metadata(detections)["role"] != "other"
    }
    foreground_started = time.perf_counter()
    writer = cv2.VideoWriter(
        str(results / "foreground-temp.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (width, height),
    )
    for frame_index, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        combined = np.zeros((height, width), dtype=bool)
        for track_id in foreground_ids:
            rle = encoded_masks[track_id].get(frame_index)
            if rle:
                combined |= decode_mask(rle).astype(bool)
        foreground = np.zeros_like(frame)
        foreground[combined] = frame[combined]
        writer.write(foreground)
    writer.release()
    run(
        [
            "ffmpeg", "-y", "-i", str(results / "foreground-temp.mp4"),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            str(results / "foreground.mp4"),
        ]
    )
    (results / "foreground-temp.mp4").unlink()
    stage_times["foreground_seconds"] = time.perf_counter() - foreground_started

    ocr_started = time.perf_counter()
    ocr_results = recognize_jersey_numbers(crop_heaps, use_gpu=device.type == "cuda")
    stage_times["ocr_seconds"] = time.perf_counter() - ocr_started
    stage_times["identify_seconds"] = time.perf_counter() - identity_started
    monitor_stop.set()
    if monitor:
        monitor.join()
    overlaps = occlusion_metrics(samples, frame_size=(width, height))
    references = payload["team_colors"]
    tracks: list[dict[str, Any]] = []
    masks_dir = results / "masks"
    masks_dir.mkdir(exist_ok=True)
    legacy_frames: dict[int, dict[str, Any]] = {}
    for track_id, track_samples in samples.items():
        if not track_samples:
            continue
        metadata = aggregate_detection_metadata(track_detections[track_id])
        if metadata["role"] == "other":
            continue
        manifest = {
            "track_id": track_id,
            "fps": FPS,
            "width": width,
            "height": height,
            "first_frame": track_samples[0]["frame"],
            "last_frame": track_samples[-1]["frame"],
            "frames": [
                {"index": frame_index, "rle": rle}
                for frame_index, rle in sorted(encoded_masks[track_id].items())
            ],
        }
        if payload.get("analysis_mode") == "auto_all":
            with gzip.open(masks_dir / f"{track_id}.json.gz", "wt", encoding="utf-8") as handle:
                json.dump(manifest, handle, separators=(",", ":"))
        else:
            for frame in manifest["frames"]:
                legacy_frames.setdefault(frame["index"], {})[str(track_id)] = frame["rle"]

        average_color = (
            tuple(int(value) for value in np.median(colors[track_id], axis=0))
            if colors[track_id]
            else None
        )
        team = (
            classify_team(average_color, references, payload["team_a"], payload["team_b"])
            if configured_fixture
            else "unknown"
        )
        role = metadata["role"]
        if configured_fixture and team == "unknown" and role == "player":
            team = classify_team(
                average_color,
                references,
                payload["team_a"],
                payload["team_b"],
                max_lab_distance=85,
            )
        if role == "referee" or team == "Referee":
            team = "Referee"
            role = "referee"
        number, identity_confidence = merge_jersey_predictions(
            ocr_results.get(track_id, (None, 0.0)),
            (metadata["jersey_number"], metadata["identity_confidence"]),
        )
        match = roster_match(payload["roster"], team, number, identity_confidence)
        if match and str(match.get("position", "")).upper() in {"GK", "GOALKEEPER"}:
            role = "goalkeeper"
        pitch_points = [tuple(sample["pitch"]) if sample["pitch"] else None for sample in track_samples]
        smoothed_points = smooth_metric_positions(pitch_points, FPS)
        speeds = speed_series(pitch_points, FPS)
        trajectory = [
            {
                **sample,
                "smoothed_pitch": list(smoothed_points[index]) if smoothed_points[index] else None,
                "speed_kmh": speeds[index],
            }
            for index, sample in enumerate(track_samples)
        ]
        valid_speeds = [speed for speed in speeds if speed is not None]
        metric_duration = (
            (track_samples[-1]["frame"] - track_samples[0]["frame"]) / FPS
            if track_samples
            else 0
        )
        tracks.append(
            {
                "object_id": track_id,
                "role": role,
                "team": team,
                "jersey_number": number,
                "player_name": match["player_name"] if match else None,
                "dominant_color": list(average_color) if average_color else None,
                "detections": [
                    {
                        "frame": int(item["frame"]),
                        "time": round(int(item["frame"]) / FPS, 3),
                        "bbox": [round(float(value), 2) for value in item["bbox"]],
                    }
                    for item in track_detections[track_id]
                ],
                "trajectory": trajectory,
                "speed_series": speeds,
                "first_frame": track_samples[0]["frame"],
                "last_frame": track_samples[-1]["frame"],
                "detector_confidence": round(
                    float(np.mean([item.get("confidence", 0) for item in track_detections[track_id]])), 3
                ),
                "auto_roster_id": match["id"] if match else None,
                "roster_id": match["id"] if match else None,
                "identity_source": "automatic" if match else "unidentified",
                "identity_confidence": identity_confidence,
                "metrics": {
                    **overlaps[track_id],
                    "mask_model_tier": os.getenv("SAM_MODEL_TIER", "large"),
                    "mask_refinement_status": "base_ready"
                    if os.getenv("SAM_MODEL_TIER", "large") == "base_plus"
                    else "large_ready",
                    "automatic_identity": {
                        "team": team,
                        "jersey_number": number,
                        "player_name": match["player_name"] if match else None,
                        "confidence": identity_confidence,
                    },
                    "distance_m": traveled_distance(pitch_points, FPS),
                    "average_speed_kmh": round(float(np.mean(valid_speeds)), 2) if valid_speeds else None,
                    "max_speed_kmh": (
                        max(valid_speeds, default=None) if metric_duration >= 1.0 else None
                    ),
                    "metric_calibration_available": bool(valid_speeds),
                    "mask_coverage_ratio": round(
                        sum(
                            int(sample["frame"]) in encoded_masks[track_id]
                            for sample in track_samples
                        )
                        / max(1, len(track_samples)),
                        3,
                    ),
                },
            }
        )

    if payload.get("analysis_mode") != "auto_all":
        legacy = {
            "fps": FPS,
            "width": width,
            "height": height,
            "frames": [
                {"index": index, "objects": legacy_frames.get(index, {})}
                for index in range(len(frame_paths))
            ],
        }
        with gzip.open(results / "masks.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(legacy, handle, separators=(",", ":"))

    (results / "tracks.json").write_text(json.dumps(tracks, ensure_ascii=False, separators=(",", ":")))
    elapsed = time.perf_counter() - started
    import torch

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
        "width": width,
        "height": height,
        "fps": FPS,
        "objects": len(tracks),
        "raw_tracker_ids": raw_track_count,
        "accepted_tracker_ids": len(track_detections),
        "maskless_tracks_rejected": len(maskless_track_ids),
        "fixture_identity_configured": configured_fixture,
        "elapsed_seconds": round(elapsed, 2),
        "effective_fps": round(len(frame_paths) / elapsed, 2),
        **{key: round(value, 2) for key, value in stage_times.items()},
        "precision": "bfloat16" if device.type == "cuda" else "float32",
        "sam_object_batch": int(os.getenv("SAM_OBJECT_BATCH", "8")),
        "sam_model_tier": os.getenv("SAM_MODEL_TIER", "large"),
        "sam_window_frames": int(os.getenv("SAM_WINDOW_FRAMES", "180")),
        "sam_window_overlap": int(os.getenv("SAM_WINDOW_OVERLAP", "30")),
        "sam_vos_optimized": os.getenv("SAM_VOS_OPTIMIZED", "true").lower() == "true",
        "sam_pad_compiled_batch": os.getenv("SAM_PAD_COMPILED_BATCH", "true").lower() == "true",
        "sam_bidirectional": os.getenv("SAM_BIDIRECTIONAL", "true").lower() == "true",
        "sam_prompts_per_track": int(os.getenv("SAM_PROMPTS_PER_TRACK", "3")),
        "detector_tracker_fps": float(os.getenv("DETECTOR_TRACKER_FPS", "10")),
        "reid_appearance_fps": float(os.getenv("REID_APPEARANCE_FPS", "5")),
        "calibration_fps": float(os.getenv("CALIBRATION_FPS", "5")),
        "role_model_enabled": os.getenv("ROLE_MODEL_ENABLED", "false").lower() == "true",
        "gsr_concat_tracklets": os.getenv("GSR_CONCAT_TRACKLETS", "false").lower() == "true",
        "sam_min_detection_iou": float(os.getenv("SAM_MIN_DETECTION_IOU", "0.1")),
        "gpu_utilization_average_percent": round(float(np.mean(gpu_utilizations)), 2) if gpu_utilizations else 0,
        "gpu_utilization_peak_percent": max(gpu_utilizations, default=0),
        "gpu_memory_used_peak_mb": max(gpu_memory_samples, default=0),
        "gpu_peak_memory_mb": gpu_peak_memory_mb,
        "gpu_peak_reserved_mb": gpu_peak_reserved_mb,
        "occlusion_events": sum(track["metrics"]["occlusion_count"] for track in tracks),
        "ids_retained": sum(bool(track["metrics"]["id_retained"]) for track in tracks),
        "active_object_frames": sum(
            int(track["last_frame"]) - int(track["first_frame"]) + 1 for track in tracks
        ),
        "dense_object_frames": len(frame_paths) * len(tracks),
        "field_association": game_state.get("association_metrics", {}),
        **tracking_metrics,
        "calibration_valid_rate": round(
            sum(item["valid"] for item in calibrations.values()) / max(1, len(calibrations)), 3
        ),
    }
    (results / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False))
    set_progress(job_dir, "upload", 90, len(tracks))
    shutil.rmtree(frames_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", type=Path)
    process(parser.parse_args().job_dir)
