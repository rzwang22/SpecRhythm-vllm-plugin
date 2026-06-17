"""Configuration defaults for the SpecRhythm research plugin."""

from dataclasses import dataclass

TARGET_VLLM_VERSION = "0.23.0"


@dataclass(frozen=True)
class SpecRhythmConfig:
    """User-facing knobs for the initial SpecRhythm prototype."""

    enabled: bool = True
    target_vllm_version: str = TARGET_VLLM_VERSION
    default_tight_tpot_ms: float = 40.0
    default_normal_tpot_ms: float = 50.0
    default_relaxed_tpot_ms: float = 150.0
    max_speculative_tokens: int = 8
