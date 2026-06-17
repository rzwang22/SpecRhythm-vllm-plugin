"""Print vLLM version and source location on the server."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path

import vllm


def main() -> None:
    print("VLLM_PLUGINS:", os.environ.get("VLLM_PLUGINS", ""))
    print("vLLM version:", importlib.metadata.version("vllm"))
    print("vLLM package:", Path(vllm.__file__).resolve().parent)


if __name__ == "__main__":
    main()
