"""Validation harness for exporting a real SGLang serving forward.

This is intentionally not a runtime integration. It proves whether the model
body, shared attention boundary, and Torch-owned serving buffers can form one
ExportedProgram without a model-architecture adapter.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(frozen=True)
class ServingExportResult:
    exported_program: Any
    report: dict[str, Any]


@dataclass(frozen=True)
class ServingMlxExecutor:
    exported_program: Any
    wrapper: torch.nn.Module
    args: tuple[Any, ...]
    execution_mode: str
    execute: Any


class ServingForwardExportWrapper(torch.nn.Module):
    """Expose changing serving metadata as a small tensor-only signature."""

    def __init__(
        self,
        model: torch.nn.Module,
        model_runner: Any,
        forward_batch: ForwardBatch,
    ) -> None:
        super().__init__()
        self.model = model
        self.forward_mode = forward_batch.forward_mode
        self.seq_lens_sum = forward_batch.seq_lens_sum
        self.extend_num_tokens = forward_batch.extend_num_tokens
        self.num_token_non_padded_cpu = forward_batch.num_token_non_padded_cpu
        backend = model_runner.attn_backend
        self.register_buffer(
            "req_to_token",
            backend.req_to_token_pool.req_to_token,
            persistent=False,
        )
        for layer_id in range(len(model_runner.attention_layers)):
            k_cache, v_cache = backend.token_to_kv_pool.get_kv_buffer(layer_id)
            self.register_buffer(f"k_cache_{layer_id}", k_cache, persistent=False)
            self.register_buffer(f"v_cache_{layer_id}", v_cache, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        extend_seq_lens: Optional[torch.Tensor],
        extend_prefix_lens: Optional[torch.Tensor],
        extend_start_loc: Optional[torch.Tensor],
        num_token_non_padded: Optional[torch.Tensor],
    ) -> torch.Tensor:
        forward_batch = ForwardBatch(
            forward_mode=self.forward_mode,
            batch_size=req_pool_indices.shape[0],
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=self.seq_lens_sum,
            positions=positions,
            extend_num_tokens=self.extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            num_token_non_padded=num_token_non_padded,
            num_token_non_padded_cpu=self.num_token_non_padded_cpu,
        )
        hidden_states = self.model.model(input_ids, positions, forward_batch)
        return torch.matmul(
            hidden_states.to(self.model.lm_head.weight.dtype),
            self.model.lm_head.weight.T,
        )


def build_serving_forward_wrapper(
    model_runner: Any,
    forward_batch: ForwardBatch,
) -> tuple[ServingForwardExportWrapper, tuple[Any, ...]]:
    wrapper = ServingForwardExportWrapper(
        model_runner.model, model_runner, forward_batch
    ).eval()
    args = (
        forward_batch.input_ids,
        forward_batch.positions,
        forward_batch.req_pool_indices,
        forward_batch.seq_lens,
        forward_batch.out_cache_loc,
        forward_batch.extend_seq_lens,
        forward_batch.extend_prefix_lens,
        forward_batch.extend_start_loc,
        forward_batch.num_token_non_padded,
    )
    return wrapper, args


def build_serving_mlx_executor(
    model_runner: Any,
    forward_batch: ForwardBatch,
) -> ServingMlxExecutor:
    """Build the MLX executor for one already-prepared serving bucket."""
    from sglang.srt.compilation.torch_compile_decoration import _to_torch
    from sglang.srt.hardware_backend.mlx.fx_backend import (
        make_mlx_decode_export_executor,
        make_mlx_prefill_export_executor,
    )

    def build_single(batch: ForwardBatch) -> ServingMlxExecutor:
        _to_torch(
            model_runner.model,
            reverse=False,
            num_tokens=batch.input_ids.shape[0],
        )
        wrapper, args = build_serving_forward_wrapper(model_runner, batch)
        exported = torch.export.export(wrapper, args, strict=True)
        if batch.forward_mode.is_decode():
            execution_mode = "decode"
            executor = make_mlx_decode_export_executor(exported, args)
        elif batch.forward_mode.is_extend():
            execution_mode = "prefill"
            executor = make_mlx_prefill_export_executor(exported, args)
        else:
            raise RuntimeError(
                f"unsupported forward mode for MLX export: {batch.forward_mode}"
            )
        return ServingMlxExecutor(
            exported_program=exported,
            wrapper=wrapper,
            args=args,
            execution_mode=execution_mode,
            execute=executor,
        )

    return build_single(forward_batch)


def export_serving_forward(
    model_runner: Any,
    forward_batch: ForwardBatch,
    report_path: str,
) -> ServingExportResult:
    """Export one real serving bucket and describe its state contract."""
    from sglang.srt.compilation.torch_compile_decoration import _to_torch
    from sglang.srt.hardware_backend.mlx.fx_backend import (
        MlxFxLoweringRegistry,
        build_mlx_fx_plan,
    )

    _to_torch(
        model_runner.model,
        reverse=False,
        num_tokens=forward_batch.input_ids.shape[0],
    )
    wrapper, args = build_serving_forward_wrapper(model_runner, forward_batch)
    exported = torch.export.export(wrapper, args, strict=True)
    graph = exported.graph_module.graph
    attention_nodes = [
        node
        for node in graph.nodes
        if "unified_attention_with_output" in str(node.target)
    ]
    mutation_outputs = [
        str(spec.target)
        for spec in exported.graph_signature.output_specs
        if "MUTATION" in str(spec.kind)
    ]
    node_targets = Counter(
        str(node.target) for node in graph.nodes if node.op == "call_function"
    )
    attention_schema = str(
        torch.ops.sglang.unified_attention_with_output.default._schema
    )
    lowering_registry = MlxFxLoweringRegistry.standard_export_decoder()
    lowering_plan = build_mlx_fx_plan(exported.graph_module, lowering_registry)
    unlifted = exported.module()
    unlifted_node_kinds = Counter(node.op for node in unlifted.graph.nodes)
    unsupported_targets = Counter(
        str(node.target) for node in lowering_plan.unsupported
    )
    report = {
        "forward_mode": str(forward_batch.forward_mode),
        "graph_nodes": len(tuple(graph.nodes)),
        "attention_nodes": len(attention_nodes),
        "attention_schema": attention_schema,
        "kv_cache_is_directly_mutable": all(
            marker in attention_schema
            for marker in ("Tensor(a15!)? k_cache", "Tensor(a16!)? v_cache")
        ),
        "mutation_outputs": mutation_outputs,
        "node_targets": dict(sorted(node_targets.items())),
        "unsupported_node_targets": dict(sorted(unsupported_targets.items())),
        "unlifted_node_kinds": dict(sorted(unlifted_node_kinds.items())),
        "unlifted_call_modules": [
            str(node.target) for node in unlifted.graph.nodes if node.op == "call_module"
        ],
        "input_specs": [
            {"kind": str(spec.kind), "target": str(spec.target)}
            for spec in exported.graph_signature.input_specs
        ],
        "output_specs": [
            {"kind": str(spec.kind), "target": str(spec.target)}
            for spec in exported.graph_signature.output_specs
        ],
    }
    if os.environ.get("SGLANG_MLX_EXPORT_EXECUTE"):
        from sglang.srt.hardware_backend.mlx.fx_backend import (
            make_mlx_decode_export_executor,
            make_mlx_prefill_export_executor,
            run_torch_decode_export_reference,
            run_torch_prefill_export_reference,
        )

        if forward_batch.forward_mode.is_decode():
            mlx_executor = make_mlx_decode_export_executor(exported, args)
            execution_mode = "decode"
        elif forward_batch.forward_mode.is_extend():
            mlx_executor = make_mlx_prefill_export_executor(exported, args)
            execution_mode = "prefill"
        else:
            mlx_executor = None
            execution_mode = "not_admitted"
        if mlx_executor is None:
            report["execution"] = {"mode": execution_mode}
            Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            return ServingExportResult(exported_program=exported, report=report)
        report["execution_inputs"] = {
            "req_pool_indices": forward_batch.req_pool_indices.cpu().tolist(),
            "seq_lens": forward_batch.seq_lens.cpu().tolist(),
            "out_cache_loc": forward_batch.out_cache_loc.cpu().tolist(),
            "extend_prefix_lens": (
                None
                if forward_batch.extend_prefix_lens is None
                else forward_batch.extend_prefix_lens.cpu().tolist()
            ),
            "extend_seq_lens": (
                None
                if forward_batch.extend_seq_lens is None
                else forward_batch.extend_seq_lens.cpu().tolist()
            ),
            "extend_start_loc": (
                None
                if forward_batch.extend_start_loc is None
                else forward_batch.extend_start_loc.cpu().tolist()
            ),
        }
        debug_attention = bool(os.environ.get("SGLANG_MLX_EXPORT_DEBUG_ATTENTION"))
        if execution_mode == "decode":
            torch_reference = run_torch_decode_export_reference(
                exported, args, return_attention=debug_attention
            )
            if debug_attention:
                torch_logits, torch_attention = torch_reference
            else:
                torch_logits = torch_reference
                torch_attention = None
        else:
            torch_reference = run_torch_prefill_export_reference(
                exported, args, return_first_attention=debug_attention
            )
            if debug_attention:
                torch_logits, torch_attention = torch_reference
            else:
                torch_logits = torch_reference
                torch_attention = None
        mlx_result = mlx_executor(*args)
        if debug_attention:
            mlx_logits, mlx_attention, mlx_first_value = mlx_result
        else:
            mlx_logits = mlx_result
            mlx_attention = None
            mlx_first_value = None
        torch.mps.synchronize()
        difference = (mlx_logits.float() - torch_logits.float()).abs()
        report["execution"] = {
            "mode": execution_mode,
            "max_abs_error": float(difference.max().cpu()),
            "mean_abs_error": float(difference.mean().cpu()),
            "allclose": bool(
                torch.allclose(
                    mlx_logits,
                    torch_logits,
                    atol=0.08,
                    rtol=0.03,
                )
            ),
        }
        if torch_attention is not None and mlx_attention is not None:
            attention_difference = (
                mlx_attention.reshape_as(torch_attention).float()
                - torch_attention.float()
            ).abs()
            per_layer_max = attention_difference.flatten(1).amax(dim=1)
            report["execution"]["first_attention_max_abs_error"] = float(
                per_layer_max[0].cpu()
            )
            report["execution"]["first_attention_mean_abs_error"] = float(
                attention_difference[0].mean().cpu()
            )
            report["execution"]["attention_max_abs_error_by_layer"] = [
                float(value.cpu()) for value in per_layer_max
            ]
            per_layer_row_max = attention_difference.flatten(2).amax(dim=2)
            report["execution"]["attention_max_abs_error_by_layer_and_row"] = [
                [float(value.cpu()) for value in row]
                for row in per_layer_row_max
            ]
            if execution_mode == "prefill" and bool(
                torch.all(forward_batch.extend_prefix_lens == 0).cpu()
            ):
                q_heads = mlx_attention.shape[2]
                kv_heads = mlx_first_value.shape[1]
                first_token_expected = mlx_first_value[0].repeat_interleave(
                    q_heads // kv_heads, dim=0
                )
                first_token_difference = (
                    mlx_attention[0, 0].float() - first_token_expected.float()
                ).abs()
                report["execution"][
                    "mlx_first_token_causal_max_abs_error"
                ] = float(first_token_difference.max().cpu())
    Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if os.environ.get("SGLANG_MLX_EXPORT_SAVE") and forward_batch.forward_mode.is_decode():
        torch.export.save(exported, report_path + ".pt2")
    return ServingExportResult(exported_program=exported, report=report)


__all__ = [
    "ServingExportResult",
    "ServingForwardExportWrapper",
    "export_serving_forward",
]
