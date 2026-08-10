"""Gating and lifecycle contract of the decode MLX region runner.

``_forward_raw`` admits any ``is_cuda_graph()`` mode to the decode runner,
so every ``can_run_graph`` rejection below is the only barrier between an
unsupported serving shape and a region executor that would silently compute
the wrong thing (spec tokens, encoder batches, LoRA-adapted weights the
exported graph never saw). Each case guards one such silent-failure path.
"""

from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.hardware_backend.mlx.region_runner import (
    _MAX_REGION_BATCH_SIZE,
    MlxRegionRunner,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")


def _make_runner(*, lora_enabled: bool = False) -> MlxRegionRunner:
    runner = MlxRegionRunner.__new__(MlxRegionRunner)
    k_cache = torch.zeros(4, 2, 8)
    pool = SimpleNamespace(
        start_layer=0,
        get_kv_buffer=lambda layer_id: (k_cache, k_cache),
    )
    runner.model_runner = SimpleNamespace(
        token_to_kv_pool=pool,
        hisparse_coordinator=None,
        model=torch.nn.Module(),
    )
    runner._lora_enabled = lora_enabled
    runner._executors = {}
    runner._failed_batch_sizes = set()
    runner._pool_token = None
    return runner


def _decode_batch(**overrides) -> SimpleNamespace:
    fields = dict(
        forward_mode=ForwardMode.DECODE,
        batch_size=1,
        spec_info=None,
        encoder_lens=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_rejects_every_non_decode_mode_that_dispatch_admits():
    # is_cuda_graph() admits these modes to the runner; only this gate
    # keeps them off the decode-shaped region.
    runner = _make_runner()
    for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.IDLE, ForwardMode.EXTEND):
        assert not runner.can_run_graph(_decode_batch(forward_mode=mode))


def test_rejects_request_shapes_the_export_never_saw():
    runner = _make_runner()
    assert not runner.can_run_graph(_decode_batch(spec_info=object()))
    assert not runner.can_run_graph(_decode_batch(encoder_lens=torch.ones(1)))
    assert not runner.can_run_graph(
        _decode_batch(capture_hidden_mode=CaptureHiddenMode.FULL)
    )
    assert not runner.can_run_graph(
        _decode_batch(batch_size=_MAX_REGION_BATCH_SIZE + 1)
    )
    assert not _make_runner(lora_enabled=True).can_run_graph(_decode_batch())


def test_failed_export_blacklists_the_batch_size_instead_of_retrying():
    runner = _make_runner()
    with mock.patch.object(
        MlxRegionRunner, "_ensure_executor", side_effect=RuntimeError("boom")
    ) as ensure:
        ensure.side_effect = None
        ensure.return_value = None
        assert not runner.can_run_graph(_decode_batch())
    runner._failed_batch_sizes.add(1)
    # A blacklisted size must short-circuit before any export attempt.
    with mock.patch.object(MlxRegionRunner, "_ensure_executor") as ensure:
        assert not runner.can_run_graph(_decode_batch())
        ensure.assert_not_called()


def test_pool_reallocation_invalidates_cached_executors():
    # Executors hold zero-copy views of the pool buffers; serving a stale
    # view after a pool reallocation would read freed storage silently.
    runner = _make_runner()
    runner._pool_token = runner._pool_identity()
    runner._executors[1] = object()
    runner._failed_batch_sizes.add(2)

    new_cache = torch.zeros(4, 2, 8)
    runner.model_runner.token_to_kv_pool.get_kv_buffer = lambda layer_id: (
        new_cache,
        new_cache,
    )
    with mock.patch(
        "sglang.srt.hardware_backend.mlx.export_validation.build_serving_mlx_executor",
        side_effect=RuntimeError("stop before real export"),
    ), mock.patch(
        "sglang.srt.hardware_backend.mlx.export_validation.serving_export_context"
    ), mock.patch(
        "sglang.srt.compilation.torch_compile_decoration._to_torch"
    ):
        runner._ensure_executor(
            _decode_batch(input_ids=torch.zeros(1, dtype=torch.int64))
        )
    assert runner._executors == {}
    assert runner._failed_batch_sizes == {1}
