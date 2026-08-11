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


def _make_model() -> torch.nn.Module:
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(8, 16, bias=False)
    model.logits_processor = SimpleNamespace(
        logit_scale=None, final_logit_softcapping=None
    )
    return model


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
        model=_make_model(),
    )
    runner._lora_enabled = lora_enabled
    runner._executors = {}
    runner._failed_batch_sizes = set()
    runner._state_token = None
    runner._constants_checked = False
    runner._model_reject_reason = None
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


def test_rejects_modes_the_region_cannot_serve():
    # Dispatch admits TARGET_VERIFY/IDLE via is_cuda_graph() and MIXED via
    # is_extend(); only this gate keeps them off the region.
    runner = _make_runner()
    for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.IDLE, ForwardMode.MIXED):
        assert not runner.can_run_graph(_decode_batch(forward_mode=mode))


def _extend_batch(**overrides):
    fields = dict(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        spec_info=None,
        encoder_lens=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        return_logprob=False,
        input_embeds=None,
        replace_embeds=None,
        input_ids=torch.zeros(300, dtype=torch.int64),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_extend_gating_rejects_shapes_the_export_never_saw():
    runner = _make_runner()
    assert not runner.can_run_graph(_extend_batch(batch_size=2))
    assert not runner.can_run_graph(_extend_batch(return_logprob=True))
    assert not runner.can_run_graph(_extend_batch(input_embeds=object()))
    assert not runner.can_run_graph(
        _extend_batch(input_ids=torch.zeros(4096, dtype=torch.int64))
    )


def test_extend_key_buckets_token_count():
    runner = _make_runner()
    assert runner._executor_key(_extend_batch()) == ("extend", 384)
    assert runner._executor_key(
        _extend_batch(input_ids=torch.zeros(512, dtype=torch.int64))
    ) == ("extend", 512)


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
    runner._failed_batch_sizes.add(("decode", 1))
    # A blacklisted size must short-circuit before any export attempt.
    with mock.patch.object(MlxRegionRunner, "_ensure_executor") as ensure:
        assert not runner.can_run_graph(_decode_batch())
        ensure.assert_not_called()


def _run_ensure_with_blocked_export(runner):
    with mock.patch(
        "sglang.srt.hardware_backend.mlx.export_validation.build_serving_mlx_executor",
        side_effect=RuntimeError("stop before real export"),
    ), mock.patch(
        "sglang.srt.hardware_backend.mlx.export_validation.serving_export_context"
    ), mock.patch(
        "sglang.srt.compilation.torch_compile_decoration._to_torch"
    ):
        runner._ensure_executor(
            _decode_batch(input_ids=torch.zeros(1, dtype=torch.int64)),
            ("decode", 1),
        )


def test_pool_reallocation_invalidates_cached_executors():
    # Executors hold zero-copy views of the pool buffers; serving a stale
    # view after a pool reallocation would read freed storage silently.
    runner = _make_runner()
    runner._state_token = runner._state_identity()
    runner._executors[("decode", 1)] = object()
    runner._failed_batch_sizes.add(("decode", 2))

    new_cache = torch.zeros(4, 2, 8)
    runner.model_runner.token_to_kv_pool.get_kv_buffer = lambda layer_id: (
        new_cache,
        new_cache,
    )
    _run_ensure_with_blocked_export(runner)
    assert runner._executors == {}
    assert runner._failed_batch_sizes == {("decode", 1)}


def test_weight_replacement_invalidates_cached_executors():
    # A weight update that REPLACES parameter storage leaves the exported
    # views aliasing the old tensors; without invalidation the region keeps
    # serving the old model silently.
    runner = _make_runner()
    runner._state_token = runner._state_identity()
    runner._executors[("decode", 1)] = object()

    runner.model_runner.model.lm_head = torch.nn.Linear(8, 16, bias=False)
    _run_ensure_with_blocked_export(runner)
    assert runner._executors == {}


def test_in_place_weight_update_keeps_executors():
    # In-place updates keep the same storage, which the views alias, so the
    # region stays correct and must not pay a re-export.
    runner = _make_runner()
    runner._state_token = runner._state_identity()
    sentinel = object()
    runner._executors[("decode", 1)] = sentinel

    with torch.no_grad():
        runner.model_runner.model.lm_head.weight.copy_(
            torch.ones_like(runner.model_runner.model.lm_head.weight)
        )
    assert runner._state_identity() == runner._state_token
    executor = runner._ensure_executor(
        _decode_batch(input_ids=torch.zeros(1, dtype=torch.int64)),
        ("decode", 1),
    )
    assert executor is sentinel


def test_nontrivial_logits_processing_disables_the_region():
    # The wrapper computes hidden @ lm_head.T; logit_scale or softcapping in
    # the real LogitsProcessor would silently diverge inside the region.
    runner = _make_runner()
    runner.model_runner.model.logits_processor.final_logit_softcapping = 30.0
    from sglang.srt.hardware_backend.mlx.region_runner import (
        _nontrivial_logits_reason,
    )

    runner._model_reject_reason = _nontrivial_logits_reason(
        runner.model_runner.model
    )
    assert runner._model_reject_reason is not None
    assert not runner.can_run_graph(_decode_batch())
