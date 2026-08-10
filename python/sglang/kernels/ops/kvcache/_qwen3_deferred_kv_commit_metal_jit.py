"""Row-wise Torch-stream commit for deferred MLX KV outputs.

MLX custom kernels cannot mutate Torch-owned input buffers.  The whole-model
MLX island therefore returns one stacked K tensor and one stacked V tensor.
The kernel chunks arbitrary layer counts to respect Metal's buffer-argument
limit instead of enqueueing one ``index_copy_`` operation per layer and pool.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from sglang.kernels.ops.attention.qwen3_mps import QWEN3_06B_METAL_SPEC

_QWEN3_NUM_LAYERS = 28
_MAX_LAYERS_PER_LAUNCH = 14
_THREADGROUP_WIDTH = 256


def _kernel_source(
    *, pool_slots: int, num_layers: int, num_kv_heads: int, head_dim: int
) -> tuple[str, tuple[tuple[str, int, int], ...]]:
    row_width = num_kv_heads * head_dim
    chunks = tuple(
        (
            f"commit_kv_layers_{start}_{min(start + _MAX_LAYERS_PER_LAUNCH, num_layers) - 1}_bf16",
            start,
            min(_MAX_LAYERS_PER_LAUNCH, num_layers - start),
        )
        for start in range(0, num_layers, _MAX_LAYERS_PER_LAUNCH)
    )

    def entry(name: str, layer_offset: int, layer_count: int) -> str:
        k_args = ",\n".join(
            f"    device bfloat* k{i} [[buffer({3 + i})]]"
            for i in range(layer_count)
        )
        v_args = ",\n".join(
            f"    device bfloat* v{i} [[buffer({3 + layer_count + i})]]"
            for i in range(layer_count)
        )
        k_cases = "\n".join(
            f"    case {i}: k{i}[destination] = new_k[source]; break;"
            for i in range(layer_count)
        )
        v_cases = "\n".join(
            f"    case {i}: v{i}[destination] = new_v[source]; break;"
            for i in range(layer_count)
        )
        return f"""
kernel void {name}(
    const device bfloat* new_k [[buffer(0)]],
    const device bfloat* new_v [[buffer(1)]],
    const device long* slots [[buffer(2)]],
{k_args},
{v_args},
    uint3 position [[thread_position_in_grid]],
    uint3 grid_size [[threads_per_grid]]) {{
  const uint row = position.x;
  const uint element = position.y;
  const uint local_layer = position.z;
  const long raw_slot = slots[row];
  if (element >= KV_ROW_WIDTH || local_layer >= LAYERS_IN_LAUNCH || raw_slot < 0 ||
      ulong(raw_slot) >= ulong(POOL_SLOTS)) {{
    return;
  }}

  const ulong source_layer = ulong(local_layer + {layer_offset});
  const ulong source =
      (source_layer * ulong(grid_size.x) + ulong(row)) * KV_ROW_WIDTH + element;
  const ulong destination = ulong(raw_slot) * KV_ROW_WIDTH + element;
  switch (local_layer) {{
{k_cases}
  }}
  switch (local_layer) {{
{v_cases}
  }}
}}
"""

    entries = "".join(
        f"#define LAYERS_IN_LAUNCH {count}\n"
        + entry(name, start, count)
        + "\n#undef LAYERS_IN_LAUNCH\n"
        for name, start, count in chunks
    )
    source = f"""
#include <metal_stdlib>
using namespace metal;

#define KV_ROW_WIDTH {row_width}
#define POOL_SLOTS {pool_slots}

{entries}
"""
    return source, chunks


@lru_cache(maxsize=16)
def _compile_library(
    pool_slots: int, num_layers: int, num_kv_heads: int, head_dim: int
):
    compile_shader = getattr(torch.mps, "compile_shader", None)
    if not callable(compile_shader):
        raise RuntimeError(
            "deferred KV commit requires torch.mps.compile_shader from "
            "Torch 2.13"
        )
    source, chunks = _kernel_source(
        pool_slots=pool_slots,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    return compile_shader(source), chunks


def warmup_qwen3_kv_commit(*, pool_slots: int) -> None:
    library, chunks = _compile_library(
        pool_slots,
        _QWEN3_NUM_LAYERS,
        QWEN3_06B_METAL_SPEC.num_kv_heads,
        QWEN3_06B_METAL_SPEC.head_dim,
    )
    for name, _, _ in chunks:
        getattr(library, name)


def _require_bf16_contiguous(name: str, tensor: torch.Tensor) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "mps"
        or tensor.dtype != torch.bfloat16
        or not tensor.is_contiguous()
    ):
        raise RuntimeError(f"{name} must be a contiguous MPS bfloat16 tensor")


def commit_deferred_kv(
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    slots: torch.Tensor,
    k_pools: list[torch.Tensor] | tuple[torch.Tensor, ...],
    v_pools: list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    """Commit stacked ``[layers, rows, KV heads, head dim]`` without host sync.

    ``num_rows`` may be a decode batch or flattened prefill token rows; each
    row is committed to the corresponding entry in ``slots``.
    """
    _require_bf16_contiguous("new_k", new_k)
    _require_bf16_contiguous("new_v", new_v)
    expected_tail = (num_kv_heads, head_dim)
    if new_k.ndim != 4 or tuple(new_k.shape[2:]) != expected_tail:
        raise RuntimeError(
            "new_k must have shape [layers, rows, KV heads, head dim], found "
            f"{tuple(new_k.shape)}"
        )
    if tuple(new_k.shape) != tuple(new_v.shape):
        raise RuntimeError(
            "new_k/new_v must have matching layer shapes, found "
            f"{tuple(new_k.shape)} and {tuple(new_v.shape)}"
        )
    num_rows = int(new_k.shape[1])
    if (
        not isinstance(slots, torch.Tensor)
        or slots.device.type != "mps"
        or slots.dtype != torch.int64
        or not slots.is_contiguous()
        or tuple(slots.shape) != (num_rows,)
    ):
        raise RuntimeError(f"slots must be contiguous MPS int64[{num_rows}]")
    num_layers = int(new_k.shape[0])
    if num_layers <= 0 or len(k_pools) != num_layers or len(v_pools) != num_layers:
        raise RuntimeError("deferred KV commit requires one K/V pool per layer")

    pool_slots = None
    for layer, (k_pool, v_pool) in enumerate(zip(k_pools, v_pools)):
        _require_bf16_contiguous(f"k_pools[{layer}]", k_pool)
        _require_bf16_contiguous(f"v_pools[{layer}]", v_pool)
        if k_pool.ndim != 3 or tuple(k_pool.shape[1:]) != expected_tail:
            raise RuntimeError(
                f"k_pools[{layer}] must use NHD layout matching the spec"
            )
        if tuple(v_pool.shape) != tuple(k_pool.shape):
            raise RuntimeError(f"K/V pool shape differs at layer {layer}")
        if pool_slots is None:
            pool_slots = int(k_pool.shape[0])
        elif int(k_pool.shape[0]) != pool_slots:
            raise RuntimeError("all layer KV pools must have the same slot count")

    if num_rows == 0:
        return
    assert pool_slots is not None
    library, chunks = _compile_library(
        pool_slots, num_layers, num_kv_heads, head_dim
    )
    group_size = (1, _THREADGROUP_WIDTH, 1)
    for name, start, count in chunks:
        end = start + count
        getattr(library, name)(
            new_k,
            new_v,
            slots,
            *k_pools[start:end],
            *v_pools[start:end],
            threads=(num_rows, num_kv_heads * head_dim, count),
            group_size=group_size,
        )


def qwen3_commit_deferred_kv(
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    slots: torch.Tensor,
    k_pools: list[torch.Tensor] | tuple[torch.Tensor, ...],
    v_pools: list[torch.Tensor] | tuple[torch.Tensor, ...],
) -> None:
    """Compatibility wrapper for the original Qwen3-0.6B entry point."""
    commit_deferred_kv(
        new_k,
        new_v,
        slots,
        k_pools,
        v_pools,
        num_kv_heads=QWEN3_06B_METAL_SPEC.num_kv_heads,
        head_dim=QWEN3_06B_METAL_SPEC.head_dim,
    )


__all__ = [
    "commit_deferred_kv",
    "qwen3_commit_deferred_kv",
    "warmup_qwen3_kv_commit",
]
