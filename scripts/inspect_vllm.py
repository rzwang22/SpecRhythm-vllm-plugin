"""Print vLLM version, source location, and likely internal patch points."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
from pathlib import Path

import vllm


MODULES_TO_CHECK = [
    "vllm",
    "vllm.config",
    "vllm.config.speculative",
    "vllm.engine.llm_engine",
    "vllm.v1.engine.llm_engine",
    "vllm.v1.core.sched.scheduler",
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.spec_decode",
    "vllm.spec_decode",
]


def main() -> None:
    print("VLLM_PLUGINS:", os.environ.get("VLLM_PLUGINS", ""))
    print("vLLM version:", importlib.metadata.version("vllm"))
    print("vLLM package:", Path(vllm.__file__).resolve().parent)
    print()
    print("Module availability:")
    for module_name in MODULES_TO_CHECK:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - inspection script
            print(f"  [missing] {module_name}: {type(exc).__name__}: {exc}")
            continue

        module_file = getattr(module, "__file__", None)
        print(f"  [ok] {module_name}: {module_file}")


if __name__ == "__main__":
    main()
