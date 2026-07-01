"""MediaPipe Face Mesh eye and coarse head-orientation estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

import config
from utils import Point, clamp, normalized_to_pixel


LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
FACE_OVAL_SAMPLE = [10, 152, 234, 454]
NOSE_TIP = 1


@dataclass
class FaceResult:
    eyes_away: bool = False
    head_away: bool = False
    yaw_ratio: float = 0.0
    pitch_ratio: float = 0.0
    face_points: List[Point] = None


class FaceAnalyzer:
    def __init__(self) -> None:
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_TRACKING_CONFIDENCE,
        )

    def process(self, frame_bgr: np.ndarray) -> FaceResult:
        h, w = frame_bgr.shape[:2]
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = self.mesh.process(rgb)
        except Exception as exc:
            print(f"Face processing skipped: {exc}")
            return FaceResult(face_points=[])

        if not result.multi_face_landmarks:
            return FaceResult(face_points=[])

        lms = result.multi_face_landmarks[0].landmark
        points = [normalized_to_pixel(lms[i].x, lms[i].y, w, h) for i in FACE_OVAL_SAMPLE + [NOSE_TIP]]
        eye_ratio = self._gaze_ratio(lms)
        yaw_ratio, pitch_ratio = self._head_ratios(lms)

        eyes_away = abs(eye_ratio) > config.EYE_AWAY_RATIO_THRESHOLD
        head_away = abs(yaw_ratio) > config.HEAD_AWAY_X_THRESHOLD or abs(pitch_ratio) > config.HEAD_AWAY_Y_THRESHOLD
        return FaceResult(
            eyes_away=eyes_away,
            head_away=head_away,
            yaw_ratio=yaw_ratio,
            pitch_ratio=pitch_ratio,
            face_points=points,
        )

    def _gaze_ratio(self, lms) -> float:
        # Iris center relative to eye corners. 0 is centered; positive/negative indicate side gaze.
        if len(lms) < 478:
            return 0.0

        def eye_offset(corner_ids, iris_ids) -> float:
            left = lms[corner_ids[0]].x
            right = lms[corner_ids[1]].x
            iris_x = float(np.mean([lms[i].x for i in iris_ids]))
            width = max(abs(right - left), 1e-6)
            return ((iris_x - min(left, right)) / width) - 0.5

        left_offset = eye_offset(LEFT_EYE, LEFT_IRIS)
        right_offset = eye_offset(RIGHT_EYE, RIGHT_IRIS)
        return float(clamp((left_offset + right_offset) / 2.0, -0.5, 0.5))

    def _head_ratios(self, lms) -> Tuple[float, float]:
        xs = [lms[i].x for i in FACE_OVAL_SAMPLE]
        ys = [lms[i].y for i in FACE_OVAL_SAMPLE]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        face_w = max(max_x - min_x, 1e-6)
        face_h = max(max_y - min_y, 1e-6)
        nose = lms[NOSE_TIP]
        yaw = ((nose.x - min_x) / face_w) - 0.5
        pitch = ((nose.y - min_y) / face_h) - 0.5
        return float(clamp(yaw, -0.5, 0.5)), float(clamp(pitch, -0.5, 0.5))

    def close(self) -> None:
        self.mesh.close()
