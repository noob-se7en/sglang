"""Exercise disk-update RPCs through a live scheduler and real MLX executors.

Run on Apple Silicon with Torch 2.13 / MLX >= 0.32:
    python test/registered/mps/test_mlx_region_weight_update.py

The tiny, local Qwen3 checkpoint has a deliberately large argmax margin. No
model download or tokenizer is required. The replacement case uses a test-only
loader hook: ordinary dense disk updates copy in place and cannot exercise
storage invalidation by themselves. Neither forward nor region dispatch is
mocked. Observers in the scheduler process prove both prefill and decode used
the region, and distinguish executor reuse from reconstruction.
"""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=90, suite="stage-b-e2e-mps")


def _observed_scheduler(*args, **kwargs):
    from sglang.srt.hardware_backend.mlx.region_runner import MlxRegionRunner
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.model_loader.loader import DefaultModelLoader

    root = Path(os.environ["SGLANG_TEST_REGION_UPDATE_DIR"])
    execute = MlxRegionRunner.execute
    load_weights = DefaultModelLoader.load_weights_and_postprocess
    previous = {}
    generations = {}

    def observed_execute(runner, batch, **kwargs):
        result = execute(runner, batch, **kwargs)
        key = runner._executor_key(batch)
        executor = runner._executors[key]
        if previous.get(key) is not executor:
            generations[key] = generations.get(key, 0) + 1
            previous[key] = executor
        with (root / "events.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "key": list(key),
                        "generation": generations[key],
                        "head_ptr": runner.model_runner.model.lm_head.weight.data_ptr(),
                    }
                )
                + "\n"
            )
        return result

    def load_with_optional_replacement(model, weights, target_device):
        marker = root / "replace-head"
        if marker.exists():
            # Replace BEFORE loading: stale executors must still see old values,
            # so a missing invalidation cannot accidentally pass this test.
            model.lm_head.weight.data = model.lm_head.weight.data.clone()
            marker.unlink()
        return load_weights(model, weights, target_device)

    with (
        patch.object(MlxRegionRunner, "execute", observed_execute),
        patch.object(
            DefaultModelLoader,
            "load_weights_and_postprocess",
            staticmethod(load_with_optional_replacement),
        ),
    ):
        run_scheduler_process(*args, **kwargs)


def _checkpoint(path, token):
    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=1024,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = Qwen3ForCausalLM(config).to(torch.bfloat16)
    with torch.no_grad():
        # Preserve nonzero Q/K/V and real attention/cache work, but make the
        # residual stream constant so the expected tokens are analytical.
        for layer in model.model.layers:
            layer.self_attn.o_proj.weight.zero_()
            layer.mlp.down_proj.weight.zero_()
        model.model.embed_tokens.weight.fill_(1)
        model.model.norm.weight.fill_(1)
        model.lm_head.weight.zero_()
        model.lm_head.weight[token].fill_(1)
    path.mkdir()
    config.architectures = ["Qwen3ForCausalLM"]
    config.save_pretrained(path)
    save_file(
        model.state_dict(), str(path / "model.safetensors"), metadata={"format": "pt"}
    )


@unittest.skipUnless(
    torch.backends.mps.is_available() and importlib.util.find_spec("mlx"),
    "requires Apple Metal and MLX",
)
class TestMlxRegionWeightUpdate(unittest.TestCase):
    def test_disk_update_reuses_views_and_replacement_reexports(self):
        from sglang.srt.entrypoints.engine import Engine

        class ObservedEngine(Engine):
            run_scheduler_process_func = staticmethod(_observed_scheduler)

        with tempfile.TemporaryDirectory(prefix="sglang-region-update-") as directory:
            root = Path(directory)
            model_a, model_b = root / "a", root / "b"
            _checkpoint(model_a, 11)
            _checkpoint(model_b, 29)
            with patch.dict(
                os.environ,
                {
                    "SGLANG_USE_MLX": "0",
                    "SGLANG_ENABLE_MLX_WHOLE_REGION": "1",
                    "SGLANG_TEST_REGION_UPDATE_DIR": directory,
                    "HF_HUB_OFFLINE": "1",
                },
            ):
                engine = ObservedEngine(
                    model_path=str(model_a),
                    device="mps",
                    dtype="bfloat16",
                    skip_tokenizer_init=True,
                    max_total_tokens=256,
                    max_running_requests=4,
                    context_length=128,
                    chunked_prefill_size=-1,
                    mem_fraction_static=0.5,
                    cuda_graph_bs_decode=[1],
                    cuda_graph_bs_prefill=[16],
                    disable_overlap_schedule=True,
                    random_seed=42,
                )
                try:

                    def generate(expected, phase):
                        event_path = root / "events.jsonl"
                        self.assertTrue(
                            event_path.exists(), "No MLX startup executions"
                        )
                        with event_path.open() as stream:
                            stream.seek(0, os.SEEK_END)
                            output = engine.generate(
                                input_ids=[3, 4, 5, 6],
                                sampling_params={
                                    "temperature": 0,
                                    "max_new_tokens": 4,
                                    "ignore_eos": True,
                                },
                            )
                            events = [json.loads(line) for line in stream]
                        self.assertEqual(output["output_ids"], [expected] * 4)
                        self.assertEqual(
                            {e["key"][0] for e in events}, {"extend", "decode"}, events
                        )
                        print(
                            json.dumps(
                                {
                                    "phase": phase,
                                    "output_ids": output["output_ids"],
                                    "region_events": events,
                                }
                            ),
                            flush=True,
                        )
                        return {e["key"][0]: e for e in events}

                    def update(path):
                        result = engine.update_weights_from_disk(str(path))
                        self.assertTrue(result[0], result)

                    baseline = generate(11, "initial")
                    update(model_b)
                    self.assertEqual(generate(29, "in_place_update"), baseline)
                    update(model_a)
                    self.assertEqual(generate(11, "in_place_restore"), baseline)

                    (root / "replace-head").touch()
                    update(model_b)
                    self.assertFalse((root / "replace-head").exists())
                    replaced = generate(29, "storage_replacement")
                    for mode in baseline:
                        self.assertNotEqual(
                            replaced[mode]["head_ptr"], baseline[mode]["head_ptr"]
                        )
                        self.assertEqual(
                            replaced[mode]["generation"],
                            baseline[mode]["generation"] + 1,
                        )
                    update(model_a)
                    self.assertEqual(generate(11, "replacement_restore"), replaced)
                finally:
                    engine.shutdown()


if __name__ == "__main__":
    unittest.main()
