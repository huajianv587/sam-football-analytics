from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np

from .protocol import LiveFrame, LiveTrack
from .stable_tracking import RawDetection, StableDetection, StableTrackRegistry
from .tracking import MotionHistory, largest_polygon, simplify_polygon


class LiveInferenceEngine:
    """One-stream A40 engine: all-person light Masks plus selected-person SAM."""

    def __init__(self) -> None:
        from ultralytics import YOLO

        self.device = self._resolve_device(os.getenv("LIVE_DEVICE", "auto"))
        # 960 keeps distant people in a broadcast frame while remaining
        # real-time on an A40. Both
        # values stay configurable for weaker edge hardware.
        self.image_size = int(os.getenv("LIVE_IMAGE_SIZE", "960"))
        self.confidence = float(os.getenv("LIVE_CONFIDENCE", "0.10"))
        self.segment_model_path = os.getenv("LIVE_SEG_MODEL", "yolo11m-seg.pt")
        self.segmenter = YOLO(self.segment_model_path)
        self.class_names = self._configured_classes()
        self.motion = MotionHistory()
        self.stable_tracks = StableTrackRegistry(
            max_missing_frames=int(os.getenv("LIVE_MAX_MISSING_FRAMES", "45"))
        )
        self.frame_times: deque[float] = deque(maxlen=30)
        self.lock = Lock()
        self.sam_predictor: Any | None = None
        self._last_refined_mask: list[tuple[float, float]] = []
        self.sam_enabled = os.getenv("LIVE_SAM_ENABLED", "true").lower() == "true" and self.device.startswith("cuda")
        if self.sam_enabled:
            self._load_sam()

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested != "auto":
            # Slurm exposes a selected GPU as "0". Ultralytics accepts it,
            # but the rest of this service needs a canonical CUDA name to
            # enable FP16 and the CUDA-only SAM refinement path.
            if requested.isdigit():
                return f"cuda:{requested}"
            return requested
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_sam(self) -> None:
        checkpoint = Path(os.getenv("LIVE_SAM_CHECKPOINT", ""))
        config = os.getenv(
            "LIVE_SAM_CONFIG", "configs/sam2.1/sam2.1_hiera_b+.yaml"
        )
        if not checkpoint.is_file():
            self.sam_enabled = False
            return
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(config, str(checkpoint), device=self.device)
        self.sam_predictor = SAM2ImagePredictor(model)

    @property
    def model_name(self) -> str:
        return Path(self.segment_model_path).name

    def _configured_classes(self) -> set[str] | None:
        """Return configured class names; None means all model classes."""
        raw = os.getenv(
            "LIVE_CLASSES",
            "person",
        ).strip()
        if not raw or raw.lower() == "all":
            return None
        return {name.strip().lower() for name in raw.split(",") if name.strip()}

    def _class_ids(self) -> list[int] | None:
        if self.class_names is None:
            return None
        names = self.segmenter.names
        if isinstance(names, list):
            names = dict(enumerate(names))
        return [int(index) for index, name in names.items() if str(name).lower() in self.class_names]

    def reset(self) -> None:
        with self.lock:
            self.motion.clear()
            self.stable_tracks.reset()
            self._last_refined_mask = []
            self.frame_times.clear()
            # Ultralytics owns tracker state on its predictor. Recreating the
            # predictor keeps unrelated camera sessions from sharing IDs.
            self.segmenter.predictor = None

    def process(
        self,
        frame_id: int,
        timestamp: float,
        jpeg: bytes,
        selected_id: int | None,
        refine_bbox: tuple[float, float, float, float] | None = None,
    ) -> LiveFrame:
        started = time.perf_counter()
        encoded = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Unable to decode the JPEG frame")
        height, width = frame.shape[:2]
        with self.lock:
            result = self.segmenter.track(
                frame,
                persist=True,
                classes=self._class_ids(),
                tracker="bytetrack.yaml",
                conf=self.confidence,
                iou=0.55,
                imgsz=self.image_size,
                device=self.device,
                half=self.device.startswith("cuda"),
                verbose=False,
            )[0]
            tracks = self._tracks(result, frame, frame_id, timestamp, selected_id, refine_bbox)
        elapsed = time.perf_counter() - started
        self.frame_times.append(elapsed)
        mean_elapsed = sum(self.frame_times) / len(self.frame_times)
        return LiveFrame(
            frame_id=frame_id,
            width=width,
            height=height,
            inference_ms=round(elapsed * 1000, 1),
            processing_fps=round(1 / mean_elapsed, 1) if mean_elapsed else 0,
            selected_id=selected_id,
            refined_mask=self._last_refined_mask,
            tracks=tracks,
        )

    def _tracks(
        self,
        result: Any,
        frame: np.ndarray,
        frame_id: int,
        timestamp: float,
        selected_id: int | None,
        refine_bbox: tuple[float, float, float, float] | None,
    ) -> list[LiveTrack]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            stable_tracks = self.stable_tracks.update(frame_id, [])
            return self._render_tracks(stable_tracks, frame, timestamp, selected_id, refine_bbox)
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else list(range(len(boxes)))
        coordinates = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        labels = boxes.cls.int().cpu().tolist() if boxes.cls is not None else [0] * len(ids)
        names = result.names
        polygons = result.masks.xy if result.masks is not None else []
        detections: list[RawDetection] = []
        for index, track_id in enumerate(ids):
            bbox = tuple(round(float(value), 1) for value in coordinates[index])
            polygon = (
                simplify_polygon(polygons[index])
                if index < len(polygons)
                else []
            )
            detections.append(RawDetection(
                raw_id=int(track_id),
                bbox=bbox,
                confidence=round(float(confidences[index]), 3),
                class_name=str(names[int(labels[index])]),
                polygon=polygon,
            ))
        stable_tracks = self.stable_tracks.update(frame_id, detections)
        return self._render_tracks(stable_tracks, frame, timestamp, selected_id, refine_bbox)

    def _render_tracks(
        self,
        stable_tracks: list[StableDetection],
        frame: np.ndarray,
        timestamp: float,
        selected_id: int | None,
        refine_bbox: tuple[float, float, float, float] | None,
    ) -> list[LiveTrack]:
        output: list[LiveTrack] = []
        active = set()
        refined_mask: list[tuple[float, float]] = []
        best_refine_id: int | None = selected_id
        if refine_bbox is not None:
            candidates = sorted(
                stable_tracks,
                key=lambda track: self._box_iou(track.bbox, refine_bbox),
                reverse=True,
            )
            if candidates and self._box_iou(candidates[0].bbox, refine_bbox) >= 0.05:
                best_refine_id = candidates[0].track_id
        for track in stable_tracks:
            active.add(track.track_id)
            motion = self.motion.update(track.track_id, timestamp, track.bbox)
            polygon = track.polygon
            source = "lightweight" if track.state == "detected" else "predicted"
            if (
                track.track_id == best_refine_id
                and track.state == "detected"
                and self.sam_predictor is not None
            ):
                refined = self._refine(frame, track.bbox)
                if refined:
                    polygon = refined
                    source = "sam"
                    refined_mask = refined
            output.append(LiveTrack(
                track_id=track.track_id,
                raw_tracker_id=track.raw_tracker_id,
                bbox=tuple(round(value, 1) for value in track.bbox),
                confidence=round(track.confidence, 3),
                class_name=track.class_name,
                mask=polygon,
                mask_source=source,
                track_state=track.state,
                first_frame=track.first_frame,
                last_seen_frame=track.last_seen_frame,
                reassociation_count=track.reassociation_count,
                trail=self.motion.trail(track.track_id),
                speed_px_s=round(motion.speed_px_s, 1),
            ))
        self.motion.retain(active)
        self._last_refined_mask = refined_mask
        return output

    @staticmethod
    def _box_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
        x1, y1 = max(first[0], second[0]), max(first[1], second[1])
        x2, y2 = min(first[2], second[2]), min(first[3], second[3])
        overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        return overlap / max(1e-6, area_first + area_second - overlap)

    def _refine(
        self, frame: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> list[tuple[float, float]]:
        import torch

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam_predictor.set_image(rgb)
            masks, scores, _ = self.sam_predictor.predict(
                box=np.asarray(bbox, dtype=np.float32), multimask_output=True
            )
        if not len(masks):
            return []
        return largest_polygon(masks[int(np.argmax(scores))])
