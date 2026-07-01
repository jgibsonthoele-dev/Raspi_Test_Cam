"""Face-state interface without a face-landmark backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from utils import Point


@dataclass
class FaceResult:
    eyes_away: bool = False
    head_away: bool = False
    yaw_ratio: float = 0.0
    pitch_ratio: float = 0.0
    face_points: List[Point] = None


class FaceAnalyzer:
    """No-op face analyzer that keeps scoring and drawing contracts stable."""

    def process(self, frame_bgr: np.ndarray) -> FaceResult:
        return FaceResult(face_points=[])

    def close(self) -> None:
        return None
