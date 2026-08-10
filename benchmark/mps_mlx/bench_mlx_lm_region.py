"""Benchmark a native mlx-lm model region with deterministic token inputs."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(percentile / 100 * (len(ordered) - 1)))
    return ordered[index]


def _time_model_call(call, reset, *, warmup: int, repeat: int):
    for _ in range(warmup):
        logits = call()
        mx.eval(logits)
        mx.synchronize()
        reset()

    latencies_ms = []
    logits = None
    for _ in range(repeat):
        mx.synchronize()
        start = time.perf_counter()
        logits = call()
        mx.eval(logits)
        mx.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000)
        reset()
    assert logits is not None
    return logits, {
        "median_ms": statistics.median(latencies_ms),
        "mean_ms": statistics.mean(latencies_ms),
        "min_ms": min(latencies_ms),
        "p90_ms": _percentile(latencies_ms, 90),
        "warmup": warmup,
        "repeat": repeat,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    token_ids = mx.array(
        rng.integers(
            0,
            10000,
            (args.batch_size, args.input_len),
            dtype=np.int32,
        )
    )
    model, _ = load(args.model_path, lazy=True)
    mx.eval(model.parameters())
    mx.synchronize()
    cache = make_prompt_cache(model)

    if args.mode == "prefill":
        call = lambda: model(token_ids, cache=cache)
        reset = lambda: [layer.trim(args.input_len) for layer in cache]
        logits, timing = _time_model_call(
            call, reset, warmup=args.warmup, repeat=args.repeat
        )
        serving_logits = logits[:, -1, :]
    else:
        prefill_logits = model(token_ids, cache=cache)
        mx.eval(prefill_logits)
        next_token_ids = mx.argmax(prefill_logits[:, -1, :], axis=-1)[:, None]
        mx.eval(next_token_ids)
        mx.synchronize()
        call = lambda: model(next_token_ids, cache=cache)
        reset = lambda: [layer.trim(1) for layer in cache]
        logits, timing = _time_model_call(
            call, reset, warmup=args.warmup, repeat=args.repeat
        )
        serving_logits = logits[:, -1, :]

    mx.eval(serving_logits)
    result = {
        "backend": "mlx-lm",
        "mode": args.mode,
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "seed": args.seed,
        "timing": timing,
        "greedy_tokens": mx.argmax(serving_logits, axis=-1).tolist(),
        "kv_cache_bytes": sum(layer.nbytes for layer in cache),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_path), **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
