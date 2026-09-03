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
from .tracking import MotionHistory, largest_polygon, simplify_polygon


class LiveInferenceEngine:
    """One-stream A40 engine: all-person light Masks plus selected-person SAM."""

    def __init__(self) -> None:
        from ultralytics import YOLO

        self.device = os.getenv("LIVE_DEVICE", "0")
        # 960 keeps distant people in a broadcast frame while remaining
        # real-time on an A40 with the small segmentation checkpoint. Both
        # values stay configurable for weaker edge hardware.
        self.image_size = int(os.getenv("LIVE_IMAGE_SIZE", "960"))
        self.confidence = float(os.getenv("LIVE_CONFIDENCE", "0.15"))
        self.segment_model_path = os.getenv("LIVE_SEG_MODEL", "yolo11s-seg.pt")
        self.segmenter = YOLO(self.segment_model_path)
        self.class_names = self._configured_classes()
        self.motion = MotionHistory()
        self.frame_times: deque[float] = deque(maxlen=30)
        self.lock = Lock()
        self.sam_predictor: Any | None = None
        self.sam_enabled = os.getenv("LIVE_SAM_ENABLED", "true").lower() == "true"
        if self.sam_enabled:
            self._load_sam()

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

        model = build_sam2(config, str(checkpoint), device="cuda")
        self.sam_predictor = SAM2ImagePredictor(model)

    @property
    def model_name(self) -> str:
        return Path(self.segment_model_path).name

    def _configured_classes(self) -> set[str] | None:
        """Return configured class names; None means all model classes."""
        raw = os.getenv(
            "LIVE_CLASSES",
            "person,cat,dog,bird,horse,sheep,cow,chair,couch,bed,dining table",
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
            self.frame_times.clear()
            # Ultralytics owns tracker state on its predictor. Recreating the
            # predictor keeps unrelated camera sessions from sharing IDs.
            self.segmenter.predictor = None

    def process(
        self, frame_id: int, timestamp: float, jpeg: bytes, selected_id: int | None
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
                half=True,
                verbose=False,
            )[0]
            tracks = self._tracks(result, frame, timestamp, selected_id)
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
            tracks=tracks,
        )

    def _tracks(
        self, result: Any, frame: np.ndarray, timestamp: float, selected_id: int | None
    ) -> list[LiveTrack]:
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            self.motion.retain(set())
            return []
        ids = boxes.id.int().cpu().tolist()
        coordinates = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        labels = boxes.cls.int().cpu().tolist() if boxes.cls is not None else [0] * len(ids)
        names = result.names
        polygons = result.masks.xy if result.masks is not None else []
        active = set(ids)
        output: list[LiveTrack] = []
        for index, track_id in enumerate(ids):
            bbox = tuple(round(float(value), 1) for value in coordinates[index])
            motion = self.motion.update(track_id, timestamp, bbox)
            polygon = (
                simplify_polygon(polygons[index])
                if index < len(polygons)
                else []
            )
            source = "lightweight"
            if selected_id == track_id and self.sam_predictor is not None:
                refined = self._refine(frame, bbox)
                if refined:
                    polygon = refined
                    source = "sam"
            output.append(LiveTrack(
                track_id=track_id,
                bbox=bbox,
                confidence=round(float(confidences[index]), 3),
                class_name=str(names[int(labels[index])]),
                mask=polygon,
                mask_source=source,
                trail=self.motion.trail(track_id),
                speed_px_s=round(motion.speed_px_s, 1),
            ))
        self.motion.retain(active)
        return output

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
