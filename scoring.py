"""Attentiveness score and temporal event tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import config
from utils import StateTimer, clamp


@dataclass
class ScoreState:
    score: int
    phone_time: float
    eyes_away_time: float
    head_away_time: float
    phone_confirmed: bool
    eyes_away: bool
    head_away: bool
    phone_penalty: int
    eye_penalty: int
    head_penalty: int


class AttentivenessScorer:
    """Applies consecutive-duration penalties to produce a 0-100 score."""

    def __init__(self) -> None:
        self.phone_timer = StateTimer()
        self.eye_timer = StateTimer()
        self.head_timer = StateTimer()

    @staticmethod
    def _table_penalty(elapsed: float, table) -> int:
        for limit, penalty in table:
            if elapsed < limit:
                return penalty
        return 0

    def update(self, phone_in_hand: bool, eyes_away: bool, head_away: bool) -> ScoreState:
        now = time.monotonic()
        raw_phone_time = self.phone_timer.update(phone_in_hand, now)
        eyes_time = self.eye_timer.update(eyes_away, now)
        head_time = self.head_timer.update(head_away, now)

        phone_confirmed = raw_phone_time >= config.PHONE_CONFIRM_SECONDS
        phone_score_time = raw_phone_time if phone_confirmed else 0.0

        phone_penalty = self._table_penalty(phone_score_time, config.PHONE_PENALTY_TABLE)
        eye_penalty = self._table_penalty(eyes_time, config.EYE_AWAY_PENALTY_TABLE)
        head_penalty = config.HEAD_AWAY_PENALTY if head_time > config.HEAD_AWAY_SECONDS else 0

        score = int(clamp(100 - phone_penalty - eye_penalty - head_penalty, 0, 100))
        return ScoreState(
            score=score,
            phone_time=phone_score_time,
            eyes_away_time=eyes_time,
            head_away_time=head_time,
            phone_confirmed=phone_confirmed,
            eyes_away=eyes_away,
            head_away=head_away,
            phone_penalty=phone_penalty,
            eye_penalty=eye_penalty,
            head_penalty=head_penalty,
        )

    def as_dict(self, state: ScoreState) -> Dict[str, float | int | bool]:
        return state.__dict__.copy()
