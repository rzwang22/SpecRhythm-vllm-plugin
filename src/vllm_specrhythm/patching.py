"""Small patching helpers for future vLLM internal hooks.

SpecRhythm will likely need version-pinned internal patches. This module keeps
that mechanism explicit instead of scattering monkey patches across files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PatchRecord:
    target: str
    attribute: str
    replacement: str


class PatchRegistry:
    """Tracks patch operations before we implement real vLLM hooks."""

    def __init__(self) -> None:
        self._records: list[PatchRecord] = []

    def add(self, target: Any, attribute: str, replacement: Any) -> None:
        self._records.append(
            PatchRecord(
                target=f"{target.__module__}.{target.__qualname__}",
                attribute=attribute,
                replacement=(
                    f"{replacement.__module__}.{replacement.__qualname__}"
                ),
            )
        )

    @property
    def records(self) -> tuple[PatchRecord, ...]:
        return tuple(self._records)
