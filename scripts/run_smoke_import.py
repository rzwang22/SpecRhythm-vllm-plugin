"""Smoke test that does not require vLLM or a GPU."""

from vllm_specrhythm.plugin import register


if __name__ == "__main__":
    register()
