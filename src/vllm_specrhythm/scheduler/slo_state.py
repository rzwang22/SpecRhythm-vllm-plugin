"""Request-level SLO state used by future scheduler patches."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SLOClass(str, Enum):
    TIGHT = "tight"
    NORMAL = "normal"
    RELAXED = "relaxed"


@dataclass
class SLOState:
    request_id: str
    slo_class: SLOClass
    target_tpot_ms: float
    delivered_tokens: int = 0
    decode_time_s: float = 0.0
    recent_acceptance_ratio: float = 1.0
    draft_confidence: float = 1.0

    @property
    def observed_tpot_ms(self) -> float:
        if self.delivered_tokens <= 0:
            return float("inf")
        return (self.decode_time_s * 1000.0) / self.delivered_tokens

    @property
    def slack(self) -> float:
        if self.target_tpot_ms <= 0:
            return float("-inf")
        return 1.0 - (self.observed_tpot_ms / self.target_tpot_ms)

    @property
    def urgency(self) -> float:
        return max(0.0, 1.0 - self.slack)

    @property
    def expected_acceptance_benefit(self) -> float:
        return self.recent_acceptance_ratio * self.draft_confidence
