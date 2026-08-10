"""Composable MLX decoder blocks for Torch-owned model execution.

The blocks operate only on MLX arrays. A model adapter supplies borrowed
weights, cache views, and architecture dimensions, so a decoder stack can stay
inside one MLX graph rather than crossing the Torch/MLX boundary per layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class DenseGqaSpec:
    """Shape contract shared by dense grouped-query decoder blocks."""

    num_q_heads: int
    num_kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class DenseGqaDecoderLayer:
    """Borrowed MLX views required by one RMSNorm/GQA/SwiGLU layer."""

    input_norm: Any
    qkv: Any
    q_norm: Any
    k_norm: Any
    rope_cache: Any
    o_proj: Any
    post_attention_norm: Any
    gate_up: Any
    down: Any
    k_pool: Any
    v_pool: Any
    input_epsilon: float
    qk_epsilon: float
    post_attention_epsilon: float


@dataclass(frozen=True)
class DenseGqaDecoderTopology:
    """Model-specific wiring expressed using reusable decoder blocks."""

    embedding: Any
    layers: tuple[DenseGqaDecoderLayer, ...]
    final_norm: Any
    final_epsilon: float
    pool_identity: int
    pool_slots: int


def rms_norm(value: Any, weight: Any, epsilon: float):
    """RMSNorm with the bf16 rounding used by the Torch serving path."""
    import mlx.core as mx

    return mx.fast.rms_norm(
        value.astype(mx.float32), weight.astype(mx.float32), epsilon
    ).astype(mx.bfloat16)


def add_rms_norm(value: Any, residual: Any, weight: Any, epsilon: float):
    """Fuse a residual add and RMSNorm while retaining the next residual."""
    import mlx.core as mx

    summed = value.astype(mx.float32) + residual.astype(mx.float32)
    normed = mx.fast.rms_norm(summed, weight.astype(mx.float32), epsilon).astype(
        mx.bfloat16
    )
    return normed, summed.astype(mx.bfloat16)


def rope_neox(value: Any, cos_sin: Any, positions: Any):
    """Apply a full-dimension NeoX rotary embedding."""
    import mlx.core as mx

    selected = mx.take(cos_sin, positions, axis=0)
    cosine, sine = mx.split(selected, 2, axis=-1)
    first, second = mx.split(value, 2, axis=-1)
    cosine = cosine[:, None, :]
    sine = sine[:, None, :]
    return mx.concatenate(
        (first * cosine - second * sine, second * cosine + first * sine), axis=-1
    ).astype(mx.bfloat16)


def run_dense_gqa_swiglu_decoder(
    topology: DenseGqaDecoderTopology,
    input_ids: Any,
    positions: Any,
    *,
    spec: DenseGqaSpec,
    attention_forward: Callable[[DenseGqaDecoderLayer, Any, Any, Any], Any],
):
    """Compose an embedding and dense decoder stack into one lazy MLX graph.

    ``attention_forward`` selects a cache policy, such as Radix decode or
    prefix-free causal prefill. It returns MLX arrays, so all intermediates
    remain in MLX until the caller exports final outputs.
    """
    import mlx.core as mx

    hidden = mx.take(topology.embedding.array, input_ids, axis=0)
    residual = None
    new_keys = []
    new_values = []
    q_width = spec.num_q_heads * spec.head_dim
    qk_width = (spec.num_q_heads + spec.num_kv_heads) * spec.head_dim

    for layer in topology.layers:
        if residual is None:
            residual = hidden
            normed = rms_norm(hidden, layer.input_norm.array, layer.input_epsilon)
        else:
            normed, residual = add_rms_norm(
                hidden, residual, layer.input_norm.array, layer.input_epsilon
            )
        qkv = normed @ mx.transpose(layer.qkv.array)
        q, k, v = mx.split(qkv, (q_width, qk_width), axis=-1)
        batch = q.shape[0]
        q = mx.fast.rms_norm(
            q.reshape(batch, spec.num_q_heads, spec.head_dim),
            layer.q_norm.array,
            layer.qk_epsilon,
        )
        k = mx.fast.rms_norm(
            k.reshape(batch, spec.num_kv_heads, spec.head_dim),
            layer.k_norm.array,
            layer.qk_epsilon,
        )
        v = mx.contiguous(v.reshape(batch, spec.num_kv_heads, spec.head_dim))
        q = rope_neox(q, layer.rope_cache.array, positions)
        k = rope_neox(k, layer.rope_cache.array, positions)

        attention = attention_forward(layer, q, k, v)
        attention = attention.reshape(batch, q_width)
        attention = attention @ mx.transpose(layer.o_proj.array)
        mlp_input, residual = add_rms_norm(
            attention,
            residual,
            layer.post_attention_norm.array,
            layer.post_attention_epsilon,
        )
        gate_up = mlp_input @ mx.transpose(layer.gate_up.array)
        gate, up = mx.split(gate_up, 2, axis=-1)
        hidden = (mx.sigmoid(gate) * gate * up) @ mx.transpose(layer.down.array)
        new_keys.append(k)
        new_values.append(v)

    hidden, _ = add_rms_norm(
        hidden, residual, topology.final_norm.array, topology.final_epsilon
    )
    return hidden, mx.stack(new_keys, axis=0), mx.stack(new_values, axis=0)
