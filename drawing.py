"""OpenCV overlay drawing for detections, pose, face cues, and HUD."""

from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

from detector import Detection
from face import FaceResult
from pose import PoseResult
from scoring import ScoreState


GREEN = (60, 220, 80)
YELLOW = (0, 210, 255)
RED = (40, 40, 230)
WHITE = (245, 245, 245)
GRAY = (150, 150, 150)
BLACK = (20, 20, 20)

POSE_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
]


def draw_detections(frame: np.ndarray, detections: Iterable[Detection]) -> None:
    for det in detections:
        x1, y1, x2, y2 = det.box
        color = RED if det.label == "phone" else GREEN
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.label} {det.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_pose(frame: np.ndarray, pose: PoseResult) -> None:
    for a, b in POSE_EDGES:
        if a in pose.landmarks and b in pose.landmarks:
            cv2.line(frame, pose.landmarks[a], pose.landmarks[b], YELLOW, 2, cv2.LINE_AA)
    for name, point in pose.landmarks.items():
        color = RED if "wrist" in name else YELLOW
        cv2.circle(frame, point, 4, color, -1, cv2.LINE_AA)


def draw_face(frame: np.ndarray, face: FaceResult) -> None:
    if not face.face_points:
        return
    for point in face.face_points:
        cv2.circle(frame, point, 2, GREEN, -1, cv2.LINE_AA)
    status = f"yaw {face.yaw_ratio:+.2f} pitch {face.pitch_ratio:+.2f}"
    cv2.putText(frame, status, (10, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, score: ScoreState, fps: float, detector_enabled: bool) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 112), BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    score_color = GREEN if score.score >= 80 else YELLOW if score.score >= 60 else RED
    cv2.putText(frame, f"SCORE {score.score}", (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.15, score_color, 3, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {fps:4.1f}", (w - 125, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)

    cv2.putText(frame, f"phone {score.phone_time:4.1f}s", (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, f"eyes away {score.eyes_away_time:4.1f}s", (180, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)

    x = 390
    if score.phone_confirmed:
        cv2.putText(frame, "PHONE DETECTED", (x, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.62, RED, 2, cv2.LINE_AA)
    if score.eyes_away:
        cv2.putText(frame, "EYES OFF ROAD", (x, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.62, RED, 2, cv2.LINE_AA)
    if not detector_enabled:
        cv2.putText(frame, "ONNX OFF", (w - 125, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2, cv2.LINE_AA)
