"""Small utilities used by the attentiveness pipeline."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


Point = Tuple[int, int]


class FPSCounter:
    """Smoothed FPS counter that avoids noisy frame-to-frame readings."""

    def __init__(self, smoothing: float = 0.9) -> None:
        self.smoothing = smoothing
        self._last = time.monotonic()
        self.fps = 0.0

    def update(self) -> float:
        now = time.monotonic()
        dt = max(now - self._last, 1e-6)
        instant = 1.0 / dt
        self.fps = instant if self.fps == 0.0 else self.smoothing * self.fps + (1.0 - self.smoothing) * instant
        self._last = now
        return self.fps


@dataclass
class StateTimer:
    """Tracks how long a boolean condition has been continuously true."""

    active_since: Optional[float] = None
    elapsed: float = 0.0

    def update(self, active: bool, now: Optional[float] = None) -> float:
        now = time.monotonic() if now is None else now
        if active:
            if self.active_since is None:
                self.active_since = now
            self.elapsed = now - self.active_since
        else:
            self.active_since = None
            self.elapsed = 0.0
        return self.elapsed


class ExpSmoother:
    """Exponential smoother for scalar values."""

    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, value: float) -> float:
        if self.value is None:
            self.value = value
        else:
            self.value = self.alpha * value + (1.0 - self.alpha) * self.value
        return self.value


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bbox_center(box: Tuple[int, int, int, int]) -> Point:
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def nearest_distance(point: Point, candidates: Iterable[Point]) -> Optional[float]:
    distances = [euclidean(point, c) for c in candidates if c is not None]
    return min(distances) if distances else None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalized_to_pixel(x: float, y: float, width: int, height: int) -> Point:
    return int(clamp(x, 0.0, 1.0) * width), int(clamp(y, 0.0, 1.0) * height)
