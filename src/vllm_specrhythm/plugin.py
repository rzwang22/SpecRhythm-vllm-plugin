"""vLLM general plugin entry point for SpecRhythm.

The first implementation deliberately avoids patching vLLM internals. It only
proves that vLLM can discover and load the out-of-tree plugin on the server.
"""

from __future__ import annotations

import logging

from vllm_specrhythm.config import TARGET_VLLM_VERSION

LOGGER = logging.getLogger(__name__)


def register() -> None:
    """Entry point loaded by vLLM through `vllm.general_plugins`."""

    message = (
        "[SpecRhythm] plugin loaded "
        f"(target vLLM version: {TARGET_VLLM_VERSION})"
    )
    LOGGER.info(message)
    print(message, flush=True)
