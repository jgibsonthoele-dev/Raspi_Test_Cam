"""MediaPipe Pose wrapper focused on upper-body driver landmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

import config
from utils import Point, normalized_to_pixel


POSE_NAMES = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}


@dataclass
class PoseResult:
    landmarks: Dict[str, Point] = field(default_factory=dict)
    visibility: Dict[str, float] = field(default_factory=dict)

    @property
    def wrists(self):
        return [p for name, p in self.landmarks.items() if name in ("left_wrist", "right_wrist")]


class PoseTracker:
    def __init__(self) -> None:
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=config.POSE_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.POSE_TRACKING_CONFIDENCE,
        )

    def process(self, frame_bgr: np.ndarray) -> PoseResult:
        h, w = frame_bgr.shape[:2]
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = self.pose.process(rgb)
        except Exception as exc:
            print(f"Pose processing skipped: {exc}")
            return PoseResult()

        if not result.pose_landmarks:
            return PoseResult()

        landmarks: Dict[str, Point] = {}
        visibility: Dict[str, float] = {}
        for name, idx in POSE_NAMES.items():
            lm = result.pose_landmarks.landmark[idx]
            if lm.visibility < 0.35:
                continue
            landmarks[name] = normalized_to_pixel(lm.x, lm.y, w, h)
            visibility[name] = float(lm.visibility)
        return PoseResult(landmarks=landmarks, visibility=visibility)

    def close(self) -> None:
        self.pose.close()
