"""Logical dual-batch state for the future SpecRhythm scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DualBatchState:
    batch_a: list[str] = field(default_factory=list)
    batch_b: list[str] = field(default_factory=list)
    verify_a_next: bool = True

    def verification_batch(self) -> list[str]:
        return self.batch_a if self.verify_a_next else self.batch_b

    def drafting_batch(self) -> list[str]:
        return self.batch_b if self.verify_a_next else self.batch_a

    def rotate(self) -> None:
        self.verify_a_next = not self.verify_a_next
