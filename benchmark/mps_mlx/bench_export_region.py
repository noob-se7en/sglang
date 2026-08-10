"""Benchmark strict-export Torch MPS vs whole-region MLX execution.

This is a validation benchmark, not a production runtime path. It prepares real
SGLang serving batches, exports the transformer body plus LM head, and times the
raw exported region without the correctness reference work used by
``SGLANG_MLX_EXPORT_VALIDATE``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sglang.benchmark.one_batch import (
    TreeCacheNamespace,
    _maybe_prepare_mlp_sync_batch,
    load_model,
    prepare_synthetic_inputs_for_latency_test,
)
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.hardware_backend.mlx.export_validation import (
    build_serving_forward_wrapper,
    build_serving_mlx_executor,
    export_serving_forward,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    compute_attention_and_moe_layers,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    set_tc_piecewise_forward_context,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

_MIN_LOGIT_COSINE = 0.999
_MAX_LOGIT_MEAN_ABS_ERROR = 0.08
_MAX_LOGIT_ABS_ERROR = 0.5
_MIN_TOP5_OVERLAP = 4
_MAX_KV_RELATIVE_L2_ERROR = 0.02
_MIN_KV_COSINE = 0.999
_MAX_KV_MEAN_ABS_ERROR = 0.05


def _synchronize(device: str) -> None:
    torch.get_device_module(device).synchronize()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1)))
    return ordered[index]


def _time_call(fn, *, device: str, warmup: int, repeat: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
        _synchronize(device)
    latencies_ms = []
    for _ in range(repeat):
        _synchronize(device)
        start = time.perf_counter()
        fn()
        _synchronize(device)
        latencies_ms.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": statistics.median(latencies_ms),
        "mean_ms": statistics.mean(latencies_ms),
        "min_ms": min(latencies_ms),
        "p90_ms": _percentile(latencies_ms, 90),
        "repeat": repeat,
        "warmup": warmup,
    }


def _ensure_export_context(model_runner: Any, forward_batch: ForwardBatch) -> Any:
    _ensure_model_layers(model_runner)
    return set_tc_piecewise_forward_context(
        forward_batch,
        model_runner.attention_layers,
        getattr(model_runner.model, "quant_config", None),
        model_runner.moe_layers,
        model_runner.moe_fusions,
        dsa_indexers=model_runner.dsa_indexers,
        mha_companion_layers=model_runner.mha_companion_layers,
        full_graph=True,
    )


def _ensure_model_layers(model_runner: Any) -> None:
    if hasattr(model_runner, "attention_layers"):
        return
    layer_model = getattr(model_runner.model, "model", model_runner.model)
    (
        model_runner.attention_layers,
        model_runner.moe_layers,
        model_runner.moe_fusions,
        model_runner.dsa_indexers,
        model_runner.mha_companion_layers,
    ) = compute_attention_and_moe_layers(layer_model)


def _prepare_prefill_batch(
    model_runner: Any, input_ids: list[np.ndarray]
) -> tuple[ScheduleBatch, ForwardBatch]:
    reqs = prepare_synthetic_inputs_for_latency_test(
        len(input_ids), max(len(row) for row in input_ids), input_ids
    )
    return _prepare_extend_batch(model_runner, reqs)


def _prepare_extend_batch(
    model_runner: Any, reqs: list[Any]
) -> tuple[ScheduleBatch, ForwardBatch]:
    dummy_tree_cache = TreeCacheNamespace(
        page_size=model_runner.server_args.page_size,
        device=model_runner.device,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
    )
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=dummy_tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    if batch.input_ids is None and getattr(batch, "prefill_input_ids_cpu", None) is not None:
        batch.input_ids = batch.prefill_input_ids_cpu.to(
            batch.device, non_blocking=True
        )
        batch.prefill_input_ids_cpu = None
    forward_batch = ForwardBatch.init_new(
        batch,
        model_runner,
        return_hidden_states_before_norm=False,
    )
    forward_batch = model_runner.eager_runner.load_batch(forward_batch)
    if forward_batch.needs_forward_metadata_init():
        if hasattr(model_runner.model, "prepare_forward_batch"):
            model_runner.model.prepare_forward_batch(forward_batch)
        model_runner.attn_backend.init_forward_metadata(forward_batch)
    return batch, forward_batch


def _prepare_prefix_reuse_batch(
    model_runner: Any,
    input_ids: list[np.ndarray],
    prefix_lens: list[int],
) -> tuple[ScheduleBatch, ForwardBatch]:
    prefix_inputs = [row[:prefix_len] for row, prefix_len in zip(input_ids, prefix_lens)]
    reqs = prepare_synthetic_inputs_for_latency_test(
        len(prefix_inputs), max(prefix_lens), prefix_inputs
    )
    _, prefix_forward_batch = _prepare_extend_batch(model_runner, reqs)
    previous_validate = os.environ.pop("SGLANG_MLX_EXPORT_VALIDATE", None)
    try:
        with torch.inference_mode():
            model_runner.forward(prefix_forward_batch)
    finally:
        if previous_validate is not None:
            os.environ["SGLANG_MLX_EXPORT_VALIDATE"] = previous_validate
    _synchronize(model_runner.device)

    req_to_token = model_runner.req_to_token_pool.req_to_token
    for req, row, prefix_len in zip(reqs, input_ids, prefix_lens):
        req.full_untruncated_fill_ids.extend(row[prefix_len:].tolist())
        req.prefix_indices = req_to_token[
            req.req_pool_idx, :prefix_len
        ].to(req.prefix_indices.dtype)
        req.logprob_start_len = -1
        req.set_extend_range(prefix_len, len(row))
    return _prepare_extend_batch(model_runner, reqs)


def _prepare_decode_batch(
    model_runner: Any, batch: ScheduleBatch, prefill_batch: ForwardBatch
) -> ForwardBatch:
    """Seed the real KV pool, then construct the next serving decode step."""
    with torch.inference_mode():
        logits_output = model_runner.forward(prefill_batch).logits_output
        next_token_ids = model_runner.sample(logits_output, prefill_batch)
    batch.input_ids = next_token_ids.to(torch.int64)
    batch.prepare_for_decode()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    forward_batch = ForwardBatch.init_new(
        batch,
        model_runner,
        return_hidden_states_before_norm=False,
    )
    forward_batch = model_runner.eager_runner.load_batch(forward_batch)
    if forward_batch.needs_forward_metadata_init():
        if hasattr(model_runner.model, "prepare_forward_batch"):
            model_runner.model.prepare_forward_batch(forward_batch)
        model_runner.attn_backend.init_forward_metadata(forward_batch)
    return forward_batch


def _snapshot_written_kv_rows(model_runner: Any, out_cache_loc: torch.Tensor):
    """Preserve only the KV rows a validation forward is allowed to mutate."""
    slots = torch.unique(out_cache_loc).to(torch.int64)
    snapshots = []
    for layer_id in range(len(model_runner.attention_layers)):
        k_pool, v_pool = model_runner.attn_backend.token_to_kv_pool.get_kv_buffer(
            layer_id
        )
        snapshots.append(
            (k_pool, v_pool, slots, k_pool[slots].clone(), v_pool[slots].clone())
        )
    return snapshots


def _restore_written_kv_rows(snapshots) -> None:
    for k_pool, v_pool, slots, saved_k, saved_v in snapshots:
        k_pool.index_copy_(0, slots, saved_k)
        v_pool.index_copy_(0, slots, saved_v)


def _kv_state_metrics(torch_state, mlx_state) -> dict[str, float]:
    if not torch_state:
        return {
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "relative_l2_error": 0.0,
            "cosine_similarity": 1.0,
        }
    device = torch_state[0][0].device
    max_error = torch.zeros((), device=device)
    absolute_error_sum = torch.zeros((), device=device)
    squared_error_sum = torch.zeros((), device=device)
    reference_squared_sum = torch.zeros((), device=device)
    candidate_squared_sum = torch.zeros((), device=device)
    dot_sum = torch.zeros((), device=device)
    element_count = 0
    for torch_layer, mlx_layer in zip(torch_state, mlx_state):
        for reference, candidate in (
            (torch_layer[3].float(), mlx_layer[3].float()),
            (torch_layer[4].float(), mlx_layer[4].float()),
        ):
            error = candidate - reference
            max_error = torch.maximum(max_error, error.abs().max())
            absolute_error_sum += error.abs().sum()
            squared_error_sum += (error * error).sum()
            reference_squared_sum += (reference * reference).sum()
            candidate_squared_sum += (candidate * candidate).sum()
            dot_sum += (reference * candidate).sum()
            element_count += reference.numel()
    relative_l2 = torch.sqrt(
        squared_error_sum / torch.clamp_min(reference_squared_sum, 1e-20)
    )
    cosine = dot_sum / torch.sqrt(
        torch.clamp_min(reference_squared_sum * candidate_squared_sum, 1e-20)
    )
    return {
        "max_abs_error": float(max_error.cpu()),
        "mean_abs_error": float((absolute_error_sum / element_count).cpu()),
        "relative_l2_error": float(relative_l2.cpu()),
        "cosine_similarity": float(cosine.cpu()),
    }


def _delta_commit_max_error(model_runner: Any, execute_mlx, slots) -> float | None:
    new_k = getattr(execute_mlx, "last_new_k", None)
    new_v = getattr(execute_mlx, "last_new_v", None)
    if new_k is None or new_v is None:
        return None
    maxima = torch.zeros(slots.numel(), device=slots.device)
    for layer_id in range(len(model_runner.attention_layers)):
        k_pool, v_pool = model_runner.attn_backend.token_to_kv_pool.get_kv_buffer(
            layer_id
        )
        layer_error = torch.maximum(
            (k_pool[slots].float() - new_k[layer_id].float()).flatten(1).amax(dim=1),
            (v_pool[slots].float() - new_v[layer_id].float()).flatten(1).amax(dim=1),
        )
        maxima = torch.maximum(maxima, layer_error)
    return float(maxima.max().cpu())


def _delta_reference_max_error(torch_state, execute_mlx, slots):
    new_k = getattr(execute_mlx, "last_new_k", None)
    new_v = getattr(execute_mlx, "last_new_v", None)
    if new_k is None or new_v is None:
        return {"k": None, "v": None}
    state_slots = torch_state[0][2]
    indices = torch.stack(
        [torch.nonzero(state_slots == slot, as_tuple=False)[0, 0] for slot in slots]
    )
    k_by_layer = []
    v_by_layer = []
    for layer_id, layer in enumerate(torch_state):
        k_by_layer.append(
            (layer[3][indices].float() - new_k[layer_id].float())
            .flatten(1)
            .amax(dim=1)
        )
        v_by_layer.append(
            (layer[4][indices].float() - new_v[layer_id].float())
            .flatten(1)
            .amax(dim=1)
        )
    k_errors = torch.stack(k_by_layer)
    v_errors = torch.stack(v_by_layer)
    return {
        "k": float(k_errors.max().cpu()),
        "v": float(v_errors.max().cpu()),
    }


@torch.inference_mode()
def _run_generation_parity(
    model_runner: Any,
    batch: ScheduleBatch,
    forward_batch: ForwardBatch,
    decode_steps: int,
) -> dict[str, Any]:
    """Compare Torch and MLX while preserving independent KV-cache histories."""
    _ensure_model_layers(model_runner)
    used_slots = torch.unique(forward_batch.out_cache_loc).to(torch.int64)
    initial_state = _snapshot_written_kv_rows(model_runner, used_slots)
    torch_state = initial_state
    mlx_state = initial_state
    steps = []
    decode_wrapper = None
    decode_executor = None

    for step in range(decode_steps + 1):
        with (
            forward_context(ForwardContext(attn_backend=model_runner.attn_backend)),
            _ensure_export_context(model_runner, forward_batch),
        ):
            current_wrapper, torch_args = build_serving_forward_wrapper(
                model_runner, forward_batch
            )
            if step > 1 and decode_wrapper is not None:
                torch_wrapper = decode_wrapper
                execute_mlx = decode_executor
            else:
                torch_wrapper = current_wrapper
                mlx_region = build_serving_mlx_executor(model_runner, forward_batch)
                execute_mlx = mlx_region.execute
                if step == 1:
                    decode_wrapper = torch_wrapper
                    decode_executor = execute_mlx
            with torch.inference_mode():
                _restore_written_kv_rows(torch_state)
                torch_logits = torch_wrapper(*torch_args)
                _synchronize(model_runner.device)
                torch_state = _snapshot_written_kv_rows(model_runner, used_slots)

                _restore_written_kv_rows(mlx_state)
                mlx_logits = execute_mlx(*torch_args)
                _synchronize(model_runner.device)
                delta_commit_error = _delta_commit_max_error(
                    model_runner, execute_mlx, forward_batch.out_cache_loc
                )
                delta_reference_error = _delta_reference_max_error(
                    torch_state, execute_mlx, forward_batch.out_cache_loc
                )
                mlx_state = _snapshot_written_kv_rows(model_runner, used_slots)

        torch_serving = _select_serving_logits(torch_logits, forward_batch)
        mlx_serving = _select_serving_logits(mlx_logits, forward_batch)
        logit_metrics = _logit_parity_metrics(torch_serving, mlx_serving)
        torch_tokens = torch.argmax(torch_serving, dim=-1)
        kv_metrics = _kv_state_metrics(torch_state, mlx_state)
        kv_valid = (
            kv_metrics["relative_l2_error"] <= _MAX_KV_RELATIVE_L2_ERROR
            and kv_metrics["cosine_similarity"] >= _MIN_KV_COSINE
            and kv_metrics["mean_abs_error"] <= _MAX_KV_MEAN_ABS_ERROR
            and delta_commit_error == 0.0
        )
        steps.append(
            {
                "phase": "prefill" if step == 0 else "decode",
                "decode_step": max(0, step - 1),
                **logit_metrics,
                "kv_state": kv_metrics,
                "kv_valid": kv_valid,
                "mlx_delta_commit_max_abs_error": delta_commit_error,
                "mlx_delta_vs_torch_max_abs_error": delta_reference_error,
                "valid": logit_metrics["valid"] and kv_valid,
            }
        )
        if step == decode_steps:
            break

        batch.input_ids = torch_tokens.to(torch.int64)
        batch.prepare_for_decode()
        _maybe_prepare_mlp_sync_batch(batch, model_runner)
        forward_batch = ForwardBatch.init_new(
            batch, model_runner, return_hidden_states_before_norm=False
        )
        forward_batch = model_runner.eager_runner.load_batch(forward_batch)
        if forward_batch.needs_forward_metadata_init():
            model_runner.attn_backend.init_forward_metadata(forward_batch)
        used_slots = torch.unique(
            torch.cat((used_slots, forward_batch.out_cache_loc.to(torch.int64)))
        )
        # Extend each branch with pristine storage for the newly allocated row.
        _restore_written_kv_rows(torch_state)
        torch_state = _snapshot_written_kv_rows(model_runner, used_slots)
        _restore_written_kv_rows(mlx_state)
        mlx_state = _snapshot_written_kv_rows(model_runner, used_slots)

    return {
        "decode_steps": decode_steps,
        "all_tokens_match": all(step["tokens_match"] for step in steps),
        "all_steps_valid": all(step["valid"] for step in steps),
        "steps": steps,
    }


def _select_serving_logits(
    logits: torch.Tensor, forward_batch: ForwardBatch
) -> torch.Tensor:
    """Select the rows consumed by SGLang's serving logits processor."""
    if forward_batch.forward_mode.is_decode():
        return logits
    if forward_batch.forward_mode.is_extend():
        last_token_indices = (
            forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
        ).to(torch.int64)
        return logits.index_select(0, last_token_indices)
    raise RuntimeError(
        f"unsupported forward mode for serving-logit validation: "
        f"{forward_batch.forward_mode}"
    )


def _logit_parity_metrics(
    torch_logits: torch.Tensor, mlx_logits: torch.Tensor
) -> dict[str, Any]:
    torch_values = torch_logits.detach().float()
    mlx_values = mlx_logits.detach().float()
    difference = (mlx_values - torch_values).abs()
    torch_top_values, torch_top_indices = torch.topk(torch_values, k=5, dim=-1)
    mlx_top_values, mlx_top_indices = torch.topk(mlx_values, k=5, dim=-1)
    torch_tokens = torch_top_indices[:, 0]
    mlx_tokens = mlx_top_indices[:, 0]
    token_match_by_row = torch_tokens == mlx_tokens
    torch_margin = torch_top_values[:, 0] - torch_top_values[:, 1]
    mlx_margin = mlx_top_values[:, 0] - mlx_top_values[:, 1]
    row_max_error = difference.amax(dim=-1)
    row_mean_error = difference.mean(dim=-1)
    top5_overlap = []
    for torch_row, mlx_row in zip(torch_top_indices.cpu(), mlx_top_indices.cpu()):
        top5_overlap.append(len(set(torch_row.tolist()) & set(mlx_row.tolist())))
    cosine = torch.nn.functional.cosine_similarity(
        torch_values, mlx_values, dim=-1
    )
    cross_top1_in_other_top5 = (
        (mlx_top_indices == torch_tokens[:, None]).any(dim=1)
        & (torch_top_indices == mlx_tokens[:, None]).any(dim=1)
    )
    near_tie = (
        ~token_match_by_row
        & cross_top1_in_other_top5
        & (torch_margin <= 2 * row_max_error)
        & (mlx_margin <= 2 * row_max_error)
    )
    valid_by_row = (
        (cosine >= _MIN_LOGIT_COSINE)
        & (row_mean_error <= _MAX_LOGIT_MEAN_ABS_ERROR)
        & (row_max_error <= _MAX_LOGIT_ABS_ERROR)
        & torch.tensor(
            [overlap >= _MIN_TOP5_OVERLAP for overlap in top5_overlap],
            device=cosine.device,
        )
        & (token_match_by_row | near_tie)
    )
    return {
        "tokens_match": bool(torch.equal(torch_tokens, mlx_tokens)),
        "tokens_match_by_row": token_match_by_row.cpu().tolist(),
        "torch_tokens": torch_tokens.cpu().tolist(),
        "mlx_tokens": mlx_tokens.cpu().tolist(),
        "torch_top1_margin": torch_margin.cpu().tolist(),
        "mlx_top1_margin": mlx_margin.cpu().tolist(),
        "top5_overlap_count": top5_overlap,
        "cross_top1_in_other_top5": cross_top1_in_other_top5.cpu().tolist(),
        "near_tie_equivalent": near_tie.cpu().tolist(),
        "cosine_similarity": cosine.cpu().tolist(),
        "max_abs_error_by_row": row_max_error.cpu().tolist(),
        "mean_abs_error_by_row": row_mean_error.cpu().tolist(),
        "max_abs_error": float(difference.max().cpu()),
        "mean_abs_error": float(difference.mean().cpu()),
        "valid_by_row": valid_by_row.cpu().tolist(),
        "valid": bool(torch.all(valid_by_row).cpu()),
    }


def _benchmark_bucket(
    model_runner: Any,
    *,
    input_lens: list[int],
    prefix_lens: list[int] | None,
    seed: int,
    mode: str,
    warmup: int,
    repeat: int,
    debug_report: str | None,
    parity_decode_steps: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    input_ids = [
        rng.integers(0, 10000, length, dtype=np.int32) for length in input_lens
    ]
    batch_size = len(input_lens)

    def prepare_batch():
        if prefix_lens is None:
            return _prepare_prefill_batch(model_runner, input_ids)
        return _prepare_prefix_reuse_batch(model_runner, input_ids, prefix_lens)

    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()
    batch, forward_batch = prepare_batch()
    if mode == "decode":
        forward_batch = _prepare_decode_batch(model_runner, batch, forward_batch)

    previous_validate = os.environ.get("SGLANG_MLX_EXPORT_VALIDATE")
    previous_delta_debug = os.environ.get("SGLANG_MLX_EXPORT_DEBUG_KV_DELTAS")
    os.environ["SGLANG_MLX_EXPORT_VALIDATE"] = "1"
    os.environ["SGLANG_MLX_EXPORT_DEBUG_KV_DELTAS"] = "1"
    try:
        generation_parity = None
        if parity_decode_steps:
            if mode != "prefill":
                raise ValueError("multi-step parity must start from prefill mode")
            generation_parity = _run_generation_parity(
                model_runner, batch, forward_batch, parity_decode_steps
            )
            model_runner.req_to_token_pool.clear()
            model_runner.token_to_kv_pool_allocator.clear()
            batch, forward_batch = prepare_batch()

        with (
            forward_context(ForwardContext(attn_backend=model_runner.attn_backend)),
            _ensure_export_context(model_runner, forward_batch),
        ):
            torch_wrapper, torch_args = build_serving_forward_wrapper(
                model_runner, forward_batch
            )
            mlx_region = build_serving_mlx_executor(model_runner, forward_batch)

            if debug_report:
                # The report uses the deferred Torch attention reference to
                # identify whether drift begins in the first MLX attention op.
                previous_execute = os.environ.get("SGLANG_MLX_EXPORT_EXECUTE")
                previous_debug = os.environ.get("SGLANG_MLX_EXPORT_DEBUG_ATTENTION")
                debug_snapshot = _snapshot_written_kv_rows(
                    model_runner, forward_batch.out_cache_loc
                )
                os.environ["SGLANG_MLX_EXPORT_EXECUTE"] = "1"
                os.environ["SGLANG_MLX_EXPORT_DEBUG_ATTENTION"] = "1"
                try:
                    export_serving_forward(model_runner, forward_batch, debug_report)
                    _synchronize(model_runner.device)
                    _restore_written_kv_rows(debug_snapshot)
                    _synchronize(model_runner.device)
                finally:
                    if previous_execute is None:
                        os.environ.pop("SGLANG_MLX_EXPORT_EXECUTE", None)
                    else:
                        os.environ["SGLANG_MLX_EXPORT_EXECUTE"] = previous_execute
                    if previous_debug is None:
                        os.environ.pop("SGLANG_MLX_EXPORT_DEBUG_ATTENTION", None)
                    else:
                        os.environ["SGLANG_MLX_EXPORT_DEBUG_ATTENTION"] = previous_debug

            def run_torch_region():
                return torch_wrapper(*torch_args)

            def run_mlx_region():
                return mlx_region.execute(*mlx_region.args)

            with torch.inference_mode():
                kv_snapshot = _snapshot_written_kv_rows(
                    model_runner, forward_batch.out_cache_loc
                )
                torch_logits = run_torch_region()
                _synchronize(model_runner.device)
                torch_kv_state = _snapshot_written_kv_rows(
                    model_runner, forward_batch.out_cache_loc
                )
                _restore_written_kv_rows(kv_snapshot)
                mlx_logits = run_mlx_region()
                _synchronize(model_runner.device)
                mlx_kv_state = _snapshot_written_kv_rows(
                    model_runner, forward_batch.out_cache_loc
                )
                direct_commit_error = _delta_commit_max_error(
                    model_runner, mlx_region.execute, forward_batch.out_cache_loc
                )
                _restore_written_kv_rows(kv_snapshot)
                _synchronize(model_runner.device)
            direct_kv_metrics = _kv_state_metrics(torch_kv_state, mlx_kv_state)
            direct_kv_valid = (
                direct_kv_metrics["relative_l2_error"]
                <= _MAX_KV_RELATIVE_L2_ERROR
                and direct_kv_metrics["cosine_similarity"] >= _MIN_KV_COSINE
                and direct_kv_metrics["mean_abs_error"]
                <= _MAX_KV_MEAN_ABS_ERROR
                and direct_commit_error == 0.0
            )
            full_difference = (mlx_logits.float() - torch_logits.float()).abs()
            torch_serving_logits = _select_serving_logits(torch_logits, forward_batch)
            mlx_serving_logits = _select_serving_logits(mlx_logits, forward_batch)
            logit_metrics = _logit_parity_metrics(
                torch_serving_logits, mlx_serving_logits
            )
            logits_allclose = bool(
                torch.allclose(
                    mlx_serving_logits,
                    torch_serving_logits,
                    atol=0.08,
                    rtol=0.03,
                )
            )
            greedy_tokens_match = logit_metrics["tokens_match"]
            with torch.inference_mode():
                torch_timing = _time_call(
                    run_torch_region,
                    device=model_runner.device,
                    warmup=warmup,
                    repeat=repeat,
                )
                mlx_timing = _time_call(
                    run_mlx_region,
                    device=model_runner.device,
                    warmup=warmup,
                    repeat=repeat,
                )
    finally:
        if previous_validate is None:
            os.environ.pop("SGLANG_MLX_EXPORT_VALIDATE", None)
        else:
            os.environ["SGLANG_MLX_EXPORT_VALIDATE"] = previous_validate
        if previous_delta_debug is None:
            os.environ.pop("SGLANG_MLX_EXPORT_DEBUG_KV_DELTAS", None)
        else:
            os.environ["SGLANG_MLX_EXPORT_DEBUG_KV_DELTAS"] = previous_delta_debug

    valid_for_performance = logit_metrics["valid"] and direct_kv_valid and (
        generation_parity is None or generation_parity["all_steps_valid"]
    )
    return {
        "batch_size": batch_size,
        "input_lens": input_lens,
        "prefix_lens": prefix_lens,
        "seed": seed,
        "tokens": sum(input_lens),
        "mode": mlx_region.execution_mode,
        "torch_mps_exported_region": torch_timing,
        "mlx_exported_region": mlx_timing,
        # Raw timing is diagnostic until serving-output parity is established.
        "raw_speedup": (
            torch_timing["median_ms"] / mlx_timing["median_ms"]
            if mlx_timing["median_ms"]
            else None
        ),
        "speedup": (
            torch_timing["median_ms"] / mlx_timing["median_ms"]
            if valid_for_performance and mlx_timing["median_ms"]
            else None
        ),
        "correctness": {
            **logit_metrics,
            "full_tensor_max_abs_error": float(full_difference.max().cpu()),
            "full_tensor_mean_abs_error": float(full_difference.mean().cpu()),
            "allclose_at_current_gate": logits_allclose,
            "greedy_tokens_match": greedy_tokens_match,
            "kv_state": direct_kv_metrics,
            "kv_commit_max_abs_error": direct_commit_error,
            "kv_valid": direct_kv_valid,
            "valid_for_performance_comparison": valid_for_performance,
            "gate": {
                "min_cosine_similarity": _MIN_LOGIT_COSINE,
                "max_mean_abs_error": _MAX_LOGIT_MEAN_ABS_ERROR,
                "max_abs_error": _MAX_LOGIT_ABS_ERROR,
                "min_top5_overlap": _MIN_TOP5_OVERLAP,
                "max_kv_relative_l2_error": _MAX_KV_RELATIVE_L2_ERROR,
                "min_kv_cosine_similarity": _MIN_KV_COSINE,
                "max_kv_mean_abs_error": _MAX_KV_MEAN_ABS_ERROR,
                "require_exact_kv_commit": True,
            },
        },
        "generation_parity": generation_parity,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-len", type=int)
    parser.add_argument(
        "--input-lens",
        type=int,
        nargs="+",
        help="Per-request prompt lengths; overrides --input-len.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prefix-lens",
        type=int,
        nargs="+",
        help="Per-request cached-prefix lengths for prefix-reuse validation.",
    )
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument(
        "--parity-decode-steps",
        type=int,
        default=0,
        help="Run untimed lockstep prefill plus this many decode steps.",
    )
    parser.add_argument(
        "--debug-report",
        help="Optional JSON path for a first-attention parity diagnostic.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/sglang-mps-mlx-export-region-bench.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_lens is not None:
        if len(args.input_lens) != args.batch_size:
            raise ValueError("--input-lens must contain one value per batch row")
        input_lens = args.input_lens
    elif args.input_len is not None:
        input_lens = [args.input_len] * args.batch_size
    else:
        raise ValueError("either --input-len or --input-lens is required")
    if any(length <= 0 for length in input_lens):
        raise ValueError("prompt lengths must be positive")
    prefix_lens = args.prefix_lens
    if prefix_lens is not None:
        if len(prefix_lens) != args.batch_size:
            raise ValueError("--prefix-lens must contain one value per batch row")
        if any(
            prefix_len <= 0 or prefix_len >= input_len
            for prefix_len, input_len in zip(prefix_lens, input_lens)
        ):
            raise ValueError("each prefix length must be between 1 and input_len - 1")
    server_args = ServerArgs.from_cli_args(args)
    _set_envs_and_config(server_args)

    port_args = PortArgs.init_new(server_args)
    runner, _ = load_model(server_args, port_args, gpu_id=0, tp_rank=0)
    model_runner = runner.torch_runner
    try:
        result = _benchmark_bucket(
            model_runner,
            input_lens=input_lens,
            prefix_lens=prefix_lens,
            seed=args.seed,
            mode=args.mode,
            warmup=args.warmup,
            repeat=args.repeat,
            debug_report=args.debug_report,
            parity_decode_steps=args.parity_decode_steps,
        )
    finally:
        with contextlib.suppress(Exception):
            model_runner.req_to_token_pool.clear()
        with contextlib.suppress(Exception):
            model_runner.token_to_kv_pool_allocator.clear()

    output = {
        "model_path": server_args.model_path,
        "device": server_args.device,
        "result": result,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "valid_for_performance_comparison": result["correctness"][
                    "valid_for_performance_comparison"
                ],
                "all_steps_valid": (
                    None
                    if result["generation_parity"] is None
                    else result["generation_parity"]["all_steps_valid"]
                ),
                "all_tokens_match": (
                    None
                    if result["generation_parity"] is None
                    else result["generation_parity"]["all_tokens_match"]
                ),
                "raw_speedup": result["raw_speedup"],
                "speedup": result["speedup"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
