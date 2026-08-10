"""Serve decode steps through the exported whole-model MLX region.

The region executor is a pure function over Torch-owned serving state: it
reads the KV pool through zero-copy views, returns next-token logits, and
commits the step's K/V delta back through the Torch-side Metal commit.
Torch keeps ownership of scheduling, pools, sampling, LoRA, and the whole
prefill path; this runner only replaces the decode forward.

One executor is exported lazily per decode batch size and reused for every
later step of that size (the region reads all step-to-step variability —
token ids, positions, cache slots, sequence lengths — from its tensor
arguments). A batch size whose export fails is blacklisted and served by
the eager Torch path instead.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sglang.srt.model_executor.runner.base_runner import BaseRunner

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Sizes beyond the cap fall back to the eager Torch path rather than paying
# an unbounded set of exports.
_MAX_REGION_BATCH_SIZE = 16


class MlxRegionRunner(BaseRunner):
    """Decode-only graph runner backed by the exported MLX region."""

    def __init__(self, model_runner: ModelRunner) -> None:
        super().__init__(model_runner)
        from sglang.srt.hardware_backend.mps.runtime import validate_mps_runtime

        validate_mps_runtime()
        self._lora_enabled = bool(model_runner.server_args.enable_lora)
        self._executors: dict[int, Any] = {}
        self._failed_batch_sizes: set[int] = set()
        self._pool_token: Optional[int] = None

    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
        if not forward_batch.forward_mode.is_decode():
            return False
        if forward_batch.spec_info is not None:
            return False
        if forward_batch.encoder_lens is not None:
            return False
        if forward_batch.capture_hidden_mode not in (None, CaptureHiddenMode.NULL):
            return False
        if self._lora_enabled or self.model_runner.hisparse_coordinator is not None:
            return False
        if forward_batch.batch_size > _MAX_REGION_BATCH_SIZE:
            return False
        if forward_batch.batch_size in self._failed_batch_sizes:
            return False
        return self._ensure_executor(forward_batch) is not None

    def load_batch(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Any = None,
        **kwargs: Any,
    ) -> ForwardBatch:
        return forward_batch

    def execute(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Any = None,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        from sglang.srt.hardware_backend.mlx.export_validation import (
            serving_forward_args,
        )

        executor = self._executors[forward_batch.batch_size]
        logits = executor.execute(*serving_forward_args(forward_batch))
        return LogitsProcessorOutput(next_token_logits=logits)

    def _pool_identity(self) -> int:
        k_cache, _ = self.model_runner.token_to_kv_pool.get_kv_buffer(
            self.model_runner.token_to_kv_pool.start_layer
        )
        return k_cache.data_ptr()

    def _ensure_executor(self, forward_batch: ForwardBatch) -> Optional[Any]:
        # A reallocated pool invalidates the zero-copy views bound at export.
        pool_token = self._pool_identity()
        if pool_token != self._pool_token:
            if self._executors:
                logger.info("MLX region: KV pool changed; re-exporting regions.")
            self._executors.clear()
            self._failed_batch_sizes.clear()
            self._pool_token = pool_token

        batch_size = forward_batch.batch_size
        executor = self._executors.get(batch_size)
        if executor is not None:
            return executor

        from sglang.srt.compilation.torch_compile_decoration import _to_torch
        from sglang.srt.hardware_backend.mlx.export_validation import (
            build_serving_mlx_executor,
            serving_export_context,
        )

        start = time.perf_counter()
        try:
            with serving_export_context(self.model_runner, forward_batch):
                region = build_serving_mlx_executor(self.model_runner, forward_batch)
        except Exception:
            logger.exception(
                "MLX region export failed for decode batch size %d; "
                "serving this size on the eager Torch path.",
                batch_size,
            )
            self._failed_batch_sizes.add(batch_size)
            return None
        finally:
            # Export switches fused ops to their compile-safe forwards;
            # restore them so the eager Torch path keeps its fast kernels.
            _to_torch(
                self.model_runner.model,
                reverse=True,
                num_tokens=forward_batch.input_ids.shape[0],
            )
        logger.info(
            "MLX region exported for decode batch size %d in %.1f s.",
            batch_size,
            time.perf_counter() - start,
        )
        self._executors[batch_size] = region
        return region
