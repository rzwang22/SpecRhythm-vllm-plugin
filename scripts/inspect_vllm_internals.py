"""Inspect vLLM internals that SpecRhythm is likely to patch.

Run this on the GPU server's vLLM environment and paste the output back into the
development thread. The script only imports modules and prints signatures; it
does not initialize CUDA or load a model.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Iterable


MODULES = [
    "vllm.config.speculative",
    "vllm.v1.core.sched.scheduler",
    "vllm.v1.core.sched.output",
    "vllm.v1.engine.llm_engine",
    "vllm.v1.engine.core",
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.spec_decode",
]

NAME_KEYWORDS = (
    "Scheduler",
    "Schedule",
    "Request",
    "Spec",
    "Draft",
    "Proposer",
    "ModelRunner",
    "Engine",
)

METHOD_KEYWORDS = (
    "schedule",
    "add",
    "update",
    "process",
    "execute",
    "propose",
    "draft",
    "spec",
    "advance",
)


def safe_signature(obj: object) -> str:
    try:
        return str(inspect.signature(obj))
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        return f"<signature unavailable: {type(exc).__name__}: {exc}>"


def interesting_name(name: str, keywords: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def describe_class(cls: type) -> None:
    print(f"  class {cls.__name__}{safe_signature(cls)}")
    for method_name, method in inspect.getmembers(cls):
        if method_name.startswith("_") and method_name not in {"__init__"}:
            continue
        if not interesting_name(method_name, METHOD_KEYWORDS):
            continue
        if inspect.isfunction(method) or inspect.ismethod(method):
            print(f"    {method_name}{safe_signature(method)}")


def describe_module(module_name: str) -> None:
    print(f"\n## {module_name}")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print(f"  import failed: {type(exc).__name__}: {exc}")
        return

    print(f"  file: {getattr(module, '__file__', '<missing>')}")
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and interesting_name(name, NAME_KEYWORDS):
            describe_class(obj)
        elif inspect.isfunction(obj) and interesting_name(name, NAME_KEYWORDS):
            print(f"  function {name}{safe_signature(obj)}")


def list_spec_decode_submodules() -> None:
    print("\n## vllm.v1.spec_decode submodules")
    try:
        package = importlib.import_module("vllm.v1.spec_decode")
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print(f"  import failed: {type(exc).__name__}: {exc}")
        return

    package_path = getattr(package, "__path__", None)
    if package_path is None:
        print("  no package path")
        return

    for module_info in pkgutil.iter_modules(package_path):
        print(f"  {module_info.name}")


def main() -> None:
    for module_name in MODULES:
        describe_module(module_name)
    list_spec_decode_submodules()


if __name__ == "__main__":
    main()
