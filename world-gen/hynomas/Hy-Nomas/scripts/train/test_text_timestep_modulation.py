"""CPU regressions for text modulation following video timesteps.

Run with: python scripts/train/test_text_timestep_modulation.py
"""

import ast
from contextlib import nullcontext
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "hymm/models/dit/nomas_cpt.py"


def assigns(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets
    )


def compile_nodes(nodes):
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            ast.ImportFrom(
                module="hymm.models.dit.modules.attention.nomas_inference_optim",
                names=[ast.alias(name="scoped_forward", asname="scoped_sparse_optim_forward")],
                level=0,
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    return compile(ast.fix_missing_locations(module), str(MODEL_PATH), "exec")


class TimeEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 4)

    def forward(self, timestep):
        return self.linear(timestep.unsqueeze(-1))


class TextRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, text, timestep, mask, **kwargs):
        self.calls.append((timestep.clone(), mask, kwargs))
        return text + timestep[:, None, None]


class TextTimestepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(MODEL_PATH.read_text())
        model = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "CPTHYNomasDiffusionTransformer"
        )
        methods = {
            node.name: node for node in model.body
            if isinstance(node, ast.FunctionDef)
        }
        forward = methods["forward"].body
        start = next(i for i, node in enumerate(forward) if assigns(node, "chunk_size"))
        end = next(i for i, node in enumerate(forward) if assigns(node, "vec_3d"))
        # Execute the production timestep section without importing GPU kernels.
        cls.modulation = compile_nodes(forward[start:end + 1])
        cls.methods = compile_nodes([
            methods["_refine_packed_text"], methods["forward_with_cache"]
        ])

    def setUp(self):
        torch.manual_seed(42)
        self.model = SimpleNamespace(time_in=TimeEmbedder())

    def prepare(self, timestep, *, prompts=2, boundaries=None, chunk_size=0,
                packed=None, sp_rank=None):
        namespace = {
            "self": self.model,
            "torch": torch,
            "t": timestep,
            "bs": timestep.shape[0],
            "th": 1,
            "tw": 2,
            "num_prompts": prompts,
            "prompt_video_boundaries": boundaries,
            "_tp_info": packed,
            "extra_kwargs": {
                "chunk_size": chunk_size,
                "packed_per_token_timestep": packed is not None,
            },
            "get_sequence_parallel_state": lambda: sp_rank is not None,
            "nccl_info": SimpleNamespace(
                sp_size=2, rank_within_spgroup=sp_rank,
                use_dynamic_ring_attention=False,
            ),
        }
        self.model.patch_size = [1, 1, 1]
        if packed is not None:
            namespace["sample_shapes"] = packed["sample_shapes"]
            namespace["img_cu"] = [0, 4, 8]
        with patch.dict(os.environ, {"LOG_TXT_VIDEO_T": ""}):
            exec(self.modulation, namespace)
        return namespace

    def assert_text_times(self, result, expected):
        torch.testing.assert_close(result["txt_t"], expected, rtol=0, atol=0)
        embedded = self.model.time_in(expected.flatten())
        if result["vec_txt"].dim() == 3:
            embedded = embedded.reshape_as(result["vec_txt"])
        torch.testing.assert_close(result["vec_txt"], embedded, rtol=0, atol=0)

    def test_scalar_timestep_shared_by_all_text_segments(self):
        timestep = torch.tensor([0.3, 0.8])
        result = self.prepare(timestep, prompts=3)
        torch.testing.assert_close(
            result["txt_t"], timestep[:, None].expand(-1, 3), rtol=0, atol=0
        )
        self.assertIs(result["vec_txt"], result["vec"])
        self.assertIsNone(result["vec_3d"])
        self.assertTrue(torch.equal(result["vec"], self.model.time_in(timestep)))

    def test_segment_diffusion_forcing(self):
        timestep = torch.tensor([[0.2, 0.2, 0.7, 0.7]])
        result = self.prepare(timestep, boundaries=[0, 2, 4])
        self.assert_text_times(result, torch.tensor([[0.2, 0.7]]))
        expected_video = result["vec_txt"].repeat_interleave(4, dim=1)
        torch.testing.assert_close(result["vec"], expected_video, rtol=0, atol=0)

    def test_frame_and_chunk_diffusion_forcing(self):
        timestep = torch.tensor([[0.2, 0.4, 0.6, 0.8]])
        for chunk_size, expected in (
            (1, timestep),
            (2, timestep.reshape(1, 2, 2).mean(dim=-1)),
        ):
            with self.subTest(chunk_size=chunk_size):
                result = self.prepare(timestep, chunk_size=chunk_size)
                self.assert_text_times(result, expected)
                expected_video = result["vec_txt"].repeat_interleave(
                    chunk_size * 2, dim=1
                )
                torch.testing.assert_close(result["vec"], expected_video)

    def test_sequence_parallel_only_shards_video_modulation(self):
        timestep = torch.tensor([[0.2, 0.2, 0.7, 0.7]])
        full = self.prepare(timestep, boundaries=[0, 2, 4])
        for rank in (0, 1):
            result = self.prepare(timestep, boundaries=[0, 2, 4], sp_rank=rank)
            torch.testing.assert_close(result["vec"], full["vec"].chunk(2, dim=1)[rank])
            torch.testing.assert_close(result["vec_txt"], full["vec_txt"])

    @staticmethod
    def packed_info():
        return {
            "sample_shapes": [(2, 1, 2), (2, 1, 2)],
            "per_sample_magi_params": [
                {"txt_tokens_per_block": 2, "num_prompts": 2,
                 "prompt_video_boundaries": [0, 2, 4]},
                {"txt_tokens_per_block": 2, "num_prompts": 1,
                 "prompt_video_boundaries": [0, 4]},
            ],
        }

    def test_packed_text_uses_each_segments_video_timestep(self):
        timestep = torch.tensor([[0.2, 0.2, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8]])
        for rank in (None, 0, 1):
            with self.subTest(sp_rank=rank):
                result = self.prepare(timestep, packed=self.packed_info(), sp_rank=rank)
                torch.testing.assert_close(
                    result["txt_t"], torch.tensor([0.2, 0.6, 0.8]), rtol=0, atol=0
                )
                expected_text = self.model.time_in(
                    torch.tensor([0.2, 0.2, 0.6, 0.6, 0.8, 0.8])
                ).unsqueeze(0)
                torch.testing.assert_close(result["vec_txt"], expected_text)
                expected_video = self.model.time_in(timestep.flatten()).unsqueeze(0)
                if rank is not None:
                    expected_video = expected_video.chunk(2, dim=1)[rank]
                torch.testing.assert_close(result["vec"], expected_video)

    def test_packed_modulation_requires_segment_boundaries(self):
        packed = {"sample_shapes": [(2, 1, 2), (2, 1, 2)]}
        with self.assertRaisesRegex(AssertionError, "per_sample_magi_params"):
            self.prepare(torch.ones(1, 8), packed=packed)

    def test_text_modulation_keeps_gradients_and_actual_zero_timesteps(self):
        timestep = torch.tensor([[0.0, 0.0, 0.7, 0.7]], requires_grad=True)
        result = self.prepare(timestep, boundaries=[0, 2, 4])
        self.assert_text_times(result, torch.tensor([[0.0, 0.7]]))
        result["vec_txt"].sum().backward()
        self.assertGreater(timestep.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.time_in.linear.weight.grad.abs().sum().item(), 0)

    def test_packed_refiner_preserves_sample_and_segment_order(self):
        namespace = {"torch": torch}
        exec(self.methods, namespace)
        refiner = TextRefiner()
        model = SimpleNamespace(txt_in=refiner, use_attention_mask=True)
        text = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
        mask = torch.ones(1, 6, dtype=torch.bool)
        info = {**self.packed_info(), "txt_token_counts": [4, 2]}
        times = torch.tensor([0.2, 0.6, 0.8])
        output, actual_mask = namespace["_refine_packed_text"](
            model, text, mask, times, info
        )
        torch.testing.assert_close(
            output, text + times.repeat_interleave(2)[None, :, None]
        )
        torch.testing.assert_close(actual_mask, mask)
        torch.testing.assert_close(refiner.calls[0][0], times)
        torch.testing.assert_close(
            refiner.calls[0][2]["seg_ids"], torch.tensor([0, 1, 0])
        )

    def test_cache_forward_uses_video_timestep_for_refiner_and_all_text_tokens(self):
        namespace = {
            "torch": torch,
            "get_sequence_parallel_state": lambda: False,
            "_prof": lambda name: nullcontext(),
        }
        exec(self.methods, namespace)
        text = torch.zeros(2, 3, 4)
        mask = torch.ones(2, 3, dtype=torch.bool)
        timestep = torch.tensor([0.2, 0.8])
        for projection in ("linear", "single_refiner"):
            for use_mask in (False, True):
                with self.subTest(projection=projection, use_mask=use_mask):
                    captured = []
                    refiner = TextRefiner()
                    def block(**kwargs):
                        captured.append(kwargs)
                        return kwargs["img"], kwargs["txt"]
                    model = SimpleNamespace(
                        patch_size=[1, 1, 1],
                        time_in=self.model.time_in,
                        img_in=lambda image: image.flatten(2).transpose(1, 2),
                        get_rotary_pos_embed=lambda *args, **kwargs: (None, None),
                        text_projection=projection,
                        txt_in=refiner if projection == "single_refiner" else nn.Identity(),
                        use_attention_mask=use_mask,
                        double_blocks=[block],
                        final_layer=lambda image, vec, **kwargs: image,
                        unpatchify=lambda image, tt, th, tw: image,
                    )
                    _, count = namespace["forward_with_cache"](
                        model, torch.zeros(2, 4, 2, 1, 1), timestep, text, mask,
                        update_cache=True,
                    )
                    self.assertEqual(count, 5)
                    self.assertIs(captured[0]["vec"], captured[0]["vec_txt"])
                    torch.testing.assert_close(
                        captured[0]["vec_txt"], model.time_in(timestep)
                    )
                    if projection == "single_refiner":
                        torch.testing.assert_close(refiner.calls[0][0], timestep)
                        self.assertIs(refiner.calls[0][1], mask if use_mask else None)
                        torch.testing.assert_close(
                            captured[0]["txt"], text + timestep[:, None, None]
                        )

    def test_public_configs_have_no_text_timestep_switch(self):
        from hydra import compose, initialize_config_dir
        from hydra.errors import ConfigCompositionException

        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
            for experiment in ("sft", "inference"):
                config = compose(config_name="config", overrides=[f"+experiment={experiment}"])
                # An unsupported timestep policy must not be a normal CLI override.
                self.assertNotIn("text_in_" + "timestep_zero", config)
                with self.assertRaises(ConfigCompositionException):
                    compose(config_name="config", overrides=[
                        f"+experiment={experiment}", "text_in_" + "timestep_zero=true"
                    ])

    def test_model_constructor_has_no_text_timestep_switch(self):
        tree = ast.parse(MODEL_PATH.read_text())
        model = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "CPTHYNomasDiffusionTransformer"
        )
        constructor = next(
            node for node in model.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        removed_option = "text_in_" + "timestep_zero"
        self.assertNotIn(
            removed_option, [arg.arg for arg in constructor.args.args]
        )
        self.assertFalse(any(
            isinstance(node, ast.Attribute) and node.attr == removed_option
            for node in ast.walk(model)
        ))
        factory = (ROOT / "hymm/models/dit/modules/__init__.py").read_text()
        self.assertNotIn(removed_option, factory)


if __name__ == "__main__":
    unittest.main()
