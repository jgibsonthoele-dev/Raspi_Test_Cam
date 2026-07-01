"""YOLOv8-pose wrapper focused on upper-body driver landmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

import config
from utils import Point

try:
    from ultralytics import YOLO
except Exception as exc:
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = exc
else:
    ULTRALYTICS_IMPORT_ERROR = None


# YOLO pose uses the COCO 17-keypoint layout.
POSE_NAMES = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
}


@dataclass
class PoseResult:
    landmarks: Dict[str, Point] = field(default_factory=dict)
    visibility: Dict[str, float] = field(default_factory=dict)

    @property
    def wrists(self):
        return [p for name, p in self.landmarks.items() if name in ("left_wrist", "right_wrist")]


class PoseTracker:
    """Runs YOLOv8-pose and returns key landmarks for the closest person."""

    def __init__(self) -> None:
        if YOLO is None:
            raise RuntimeError(
                "Ultralytics could not be imported, so YOLOv8-pose is unavailable. "
                f"Import error: {ULTRALYTICS_IMPORT_ERROR!r}. Install with `pip install ultralytics`."
            )

        model_path = Path(config.POSE_MODEL_PATH)
        self.model_name = str(model_path if model_path.exists() and model_path.stat().st_size > 0 else config.POSE_MODEL_FALLBACK)
        self.model = YOLO(self.model_name)

    def process(self, frame_bgr: np.ndarray) -> PoseResult:
        try:
            # Ultralytics accepts BGR numpy arrays. verbose=False avoids terminal spam on the Pi.
            results = self.model.predict(
                frame_bgr,
                imgsz=config.POSE_IMAGE_SIZE,
                conf=config.POSE_CONFIDENCE,
                device=config.POSE_DEVICE,
                verbose=False,
            )
        except Exception as exc:
            print(f"YOLOv8-pose processing skipped: {exc}")
            return PoseResult()

        if not results:
            return PoseResult()

        result = results[0]
        if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
            return PoseResult()

        person_index = self._select_driver_index(result)
        xy = result.keypoints.xy[person_index].cpu().numpy()
        conf = result.keypoints.conf[person_index].cpu().numpy() if result.keypoints.conf is not None else None

        landmarks: Dict[str, Point] = {}
        visibility: Dict[str, float] = {}
        h, w = frame_bgr.shape[:2]
        for name, idx in POSE_NAMES.items():
            if idx >= len(xy):
                continue
            score = float(conf[idx]) if conf is not None else 1.0
            x, y = float(xy[idx][0]), float(xy[idx][1])
            if score < config.POSE_KEYPOINT_CONFIDENCE or x <= 0 or y <= 0:
                continue
            landmarks[name] = (int(max(0, min(w - 1, x))), int(max(0, min(h - 1, y))))
            visibility[name] = score

        return PoseResult(landmarks=landmarks, visibility=visibility)

    @staticmethod
    def _select_driver_index(result) -> int:
        # In a cabin view the driver is normally the largest/closest person.
        if result.boxes is None or result.boxes.xyxy is None or len(result.boxes.xyxy) == 0:
            return 0
        boxes = result.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return int(np.argmax(areas))

    def close(self) -> None:
        # Ultralytics does not require explicit model shutdown.
        return None
