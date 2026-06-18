"""Verify that vLLM discovers and loads the SpecRhythm general plugin."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path


def main() -> None:
    os.environ.setdefault("VLLM_PLUGINS", "specrhythm")

    import vllm  # noqa: PLC0415
    from vllm.plugins import load_general_plugins  # noqa: PLC0415

    load_general_plugins()

    print("VLLM_PLUGINS:", os.environ.get("VLLM_PLUGINS", ""))
    print("vLLM version:", importlib.metadata.version("vllm"))
    print("vLLM package:", Path(vllm.__file__).resolve().parent)


if __name__ == "__main__":
    main()
