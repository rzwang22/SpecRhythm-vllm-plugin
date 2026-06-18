"""Run a small vLLM draft-model speculative decoding smoke test."""

from __future__ import annotations

import argparse
import os
from time import perf_counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-model",
        required=True,
        help="Target model path or HF id.",
    )
    parser.add_argument(
        "--draft-model",
        required=True,
        help="Draft model path or HF id.",
    )
    parser.add_argument("--target-tp", type=int, default=1)
    parser.add_argument("--draft-tp", type=int, default=1)
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--prompt",
        default="Write a Python function that reverses a list.",
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("VLLM_PLUGINS", "specrhythm")
    args = parse_args()

    from vllm import LLM, SamplingParams  # noqa: PLC0415

    speculative_config = {
        "method": "draft_model",
        "model": args.draft_model,
        "draft_tensor_parallel_size": args.draft_tp,
        "num_speculative_tokens": args.num_speculative_tokens,
    }

    llm_kwargs = {
        "model": args.target_model,
        "tensor_parallel_size": args.target_tp,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
        "speculative_config": speculative_config,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    start = perf_counter()
    llm = LLM(**llm_kwargs)
    load_s = perf_counter() - start

    start = perf_counter()
    outputs = llm.generate([args.prompt], sampling_params)
    generate_s = perf_counter() - start

    text = outputs[0].outputs[0].text
    token_ids = outputs[0].outputs[0].token_ids

    print("Speculative smoke succeeded")
    print(f"model_load_s={load_s:.3f}")
    print(f"generate_s={generate_s:.3f}")
    print(f"output_tokens={len(token_ids)}")
    print("output:")
    print(text)


if __name__ == "__main__":
    main()
