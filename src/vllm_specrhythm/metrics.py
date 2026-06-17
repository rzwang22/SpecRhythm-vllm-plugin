"""Metrics helpers for heterogeneous SLO serving experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestMetrics:
    request_id: str
    output_tokens: int
    decode_time_s: float
    tpot_slo_ms: float

    @property
    def tpot_ms(self) -> float:
        if self.output_tokens <= 0:
            return float("inf")
        return (self.decode_time_s * 1000.0) / self.output_tokens

    @property
    def slo_attained(self) -> bool:
        return self.tpot_ms <= self.tpot_slo_ms


def goodput_tokens_per_s(
    requests: list[RequestMetrics],
    measurement_time_s: float,
) -> float:
    if measurement_time_s <= 0:
        return 0.0
    useful_tokens = sum(
        request.output_tokens for request in requests if request.slo_attained
    )
    return useful_tokens / measurement_time_s


def slo_attainment(requests: list[RequestMetrics]) -> float:
    if not requests:
        return 0.0
    attained = sum(1 for request in requests if request.slo_attained)
    return attained / len(requests)
