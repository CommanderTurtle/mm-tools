"""CPU regressions for fixed wide LoRA coverage and adapter/checkpoint reload.

Run with: PYTHONPATH=. python scripts/train/test_lora_targets.py
"""

import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)

from hymm.models.checkpointing.lora_checkpoint import (
    _copy_optimizer_state,
    load_lora_training_checkpoint,
    save_lora_training_checkpoint,
    shutdown_lora_checkpoint_writer,
    wait_for_pending_lora_checkpoint,
)
from hymm.models.lora import (
    LORA_TARGETS,
    LoRALinear,
    inject_lora,
    load_lora_adapter,
    save_lora_adapter,
)
from hymm.utils.helpers import ScalarStates


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        for branch in ("img", "txt"):
            for projection in ("q", "k", "v", "proj"):
                setattr(self, f"{branch}_attn_{projection}", nn.Linear(8, 8))
            setattr(
                self, f"{branch}_mlp",
                nn.ModuleDict({"fc1": nn.Linear(8, 16), "fc2": nn.Linear(16, 8)}),
            )
            setattr(
                self, f"{branch}_mod", nn.ModuleDict({"linear": nn.Linear(8, 8)})
            )

    def forward(self, x):
        outputs = []
        for branch in ("img", "txt"):
            attention = sum(
                getattr(self, f"{branch}_attn_{p}")(x)
                for p in ("q", "k", "v", "proj")
            )
            mlp = getattr(self, f"{branch}_mlp")
            outputs.append(
                attention + mlp["fc2"](torch.tanh(mlp["fc1"](x)))
                + getattr(self, f"{branch}_mod")["linear"](x)
            )
        return sum(outputs)


class ToyCamera(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_attn_q = nn.Linear(8, 8)
        self.keyboard_attn_kv = nn.Linear(8, 16)
        self.proj_keyboard = nn.Linear(8, 8)

    def forward(self, x):
        key, value = self.keyboard_attn_kv(x).chunk(2, dim=-1)
        return self.img_attn_q(x) + self.proj_keyboard(key + value)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.double_blocks = nn.ModuleList([ToyBlock()])
        self.action_modules = nn.ModuleList([ToyCamera()])
        self.final_layer = nn.ModuleDict({"linear": nn.Linear(8, 8)})
        # Same leaf names outside the supported blocks must not get adapters.
        self.txt_in = ToyBlock()
        self.img_in = nn.Linear(8, 8)

    def forward(self, x):
        return self.final_layer["linear"](
            self.double_blocks[0](x) + self.action_modules[0](x)
        )


class LoRATargetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_fixed_wide_scope(self):
        model = ToyModel()
        replaced = inject_lora(model, rank=2, alpha=2)
        self.assertEqual(len(replaced), 17)
        for branch in ("img", "txt"):
            for suffix in (
                "attn_q", "attn_k", "attn_v", "attn_proj",
                "mlp.fc1", "mlp.fc2", "mod.linear",
            ):
                self.assertIn(f"double_blocks.0.{branch}_{suffix}", replaced)
        self.assertTrue(all(
            name.startswith(("double_blocks.", "action_modules."))
            for name in replaced
        ))
        self.assertIsInstance(model.txt_in.img_mlp["fc1"], nn.Linear)
        self.assertIsInstance(model.final_layer["linear"], nn.Linear)
        self.assertIsInstance(model.img_in, nn.Linear)

    def test_wide_gradients_and_adapter_reload(self):
        model = ToyModel()
        base = copy.deepcopy(model)
        inputs = torch.randn(4, 8)
        replaced = inject_lora(model, rank=2, alpha=2)
        self.assertEqual(len(replaced), 17)
        self.assertIsInstance(model.final_layer["linear"], nn.Linear)
        torch.testing.assert_close(model(inputs), base(inputs), rtol=0, atol=0)
        self.assertTrue(all(
            p.requires_grad == name.endswith(("lora_A.weight", "lora_B.weight"))
            for name, p in model.named_parameters()
        ))

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(inputs).square().mean().backward()
            for name in replaced:
                layer = model.get_submodule(name)
                self.assertGreater(layer.lora_B.weight.grad.norm().item(), 0)
                if step == 1:
                    self.assertGreater(layer.lora_A.weight.grad.norm().item(), 0)
                self.assertIsNone(layer.base.weight.grad)
            optimizer.step()

        with tempfile.TemporaryDirectory(prefix="hy_lora_targets_") as directory:
            save_lora_adapter(model, directory, rank=2, alpha=2, dropout=0)
            metadata = json.loads((Path(directory) / "adapter_config.json").read_text())
            self.assertEqual(metadata["targets"], list(LORA_TARGETS))
            config = load_lora_adapter(base, Path(directory))
            self.assertEqual(config, {"rank": 2, "alpha": 2, "dropout": 0})
            torch.testing.assert_close(model(inputs), base(inputs), rtol=0, atol=0)

    def test_checkpoint_wrapped_path_matching(self):
        model = ToyModel()
        wrapper = nn.Module()
        wrapper.add_module("_checkpoint_wrapped_module", model.double_blocks[0])
        model.double_blocks[0] = wrapper
        names = inject_lora(model, rank=2, alpha=2)
        self.assertEqual(len(names), 17)
        name = "double_blocks.0._checkpoint_wrapped_module.img_mod.linear"
        self.assertIn(name, names)
        self.assertIsInstance(model.get_submodule(name), LoRALinear)

    def test_target_override_api_removed(self):
        self.assertNotIn("targets", inspect.signature(inject_lora).parameters)
        self.assertNotIn("targets", inspect.signature(save_lora_adapter).parameters)
        with self.assertRaises(TypeError):
            inject_lora(ToyModel(), targets=["img_attn_q"])
        with self.assertRaisesRegex(ValueError, "no supported wide"):
            inject_lora(nn.Linear(8, 8))

    def test_target_override_config_removed(self):
        from hydra import compose, initialize_config_dir
        from hydra.errors import ConfigCompositionException

        config_dir = Path(__file__).resolve().parents[2] / "configs"
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
            cfg = compose(config_name="config")
            self.assertNotIn("targets", cfg.training.lora)
            with self.assertRaises(ConfigCompositionException):
                compose(config_name="config", overrides=["training.lora.targets=[]"])

    def test_metadata_cannot_select_attention_only_scope(self):
        model = ToyModel()
        inject_lora(model, rank=2, alpha=2)
        with tempfile.TemporaryDirectory(prefix="hy_lora_legacy_") as directory:
            directory = Path(directory)
            save_lora_adapter(model, directory, rank=2, alpha=2, dropout=0)
            metadata = json.loads((directory / "adapter_config.json").read_text())
            # A legacy/custom field must not change which layers get injected.
            metadata["targets"] = ["img_attn_q"]
            (directory / "adapter_config.json").write_text(json.dumps(metadata))
            restored = ToyModel()
            load_lora_adapter(restored, directory)
            self.assertEqual(
                sum(isinstance(m, LoRALinear) for m in restored.modules()), 17
            )

            # Old attention-only tensors cannot silently initialize missing MLP
            # and modulation adapters when loaded under the fixed policy.
            from safetensors.torch import load_file, save_file

            tensors = load_file(str(directory / "adapter.safetensors"))
            tensors = {k: v for k, v in tensors.items() if ".img_attn_q." in k}
            save_file(tensors, str(directory / "adapter.safetensors"))
            with self.assertRaisesRegex(ValueError, "fixed wide LoRA scope"):
                load_lora_adapter(ToyModel(), directory)

    def test_adamw_resume_with_unused_targets(self):
        model = ToyModel()
        model.double_blocks.append(ToyBlock())  # Not used in forward().
        inject_lora(model, rank=2, alpha=2)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-3)
        model(torch.randn(4, 8)).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _, saved = get_state_dict(model, optimizer)
        saved = copy.deepcopy(saved)
        self.assertNotIn("double_blocks.1.img_mlp.fc1.lora_A.weight", saved["state"])

        # Loading both an already-used optimizer and a fresh optimizer must
        # accept target parameters for which no AdamW moments were ever saved.
        for resumed in (optimizer, torch.optim.AdamW(params, lr=1e-3)):
            _, live = get_state_dict(model, resumed)
            _copy_optimizer_state(live, saved)
            set_state_dict(
                model, resumed, model_state_dict={}, optim_state_dict=live,
                options=StateDictOptions(strict=False),
            )
            _, restored = get_state_dict(model, resumed)
            self.assertEqual(set(restored["state"]), set(saved["state"]))
            for name, state in saved["state"].items():
                for field, value in state.items():
                    torch.testing.assert_close(restored["state"][name][field], value)
            self.assertEqual(restored["param_groups"], saved["param_groups"])

    def test_training_checkpoint_uses_fixed_scope(self):
        from torch.optim.lr_scheduler import LambdaLR
        from scripts.train.export_lora_checkpoint import export_adapter

        model = ToyModel()
        base = copy.deepcopy(model)
        inject_lora(model, rank=2, alpha=2)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad]
        )
        scheduler = LambdaLR(optimizer, lambda _: 1)
        inputs = torch.randn(4, 8)
        model(inputs).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        expected = model(inputs).detach()
        with tempfile.TemporaryDirectory(prefix="hy_lora_checkpoint_") as directory:
            try:
                config = {"rank": 2, "alpha": 2, "dropout": 0}
                save_lora_training_checkpoint(
                    model, optimizer, scheduler, directory,
                    ScalarStates(train_steps=1, update_steps=1), config,
                )
                wait_for_pending_lora_checkpoint()
                checkpoint = Path(directory) / "checkpoints/global_step-1"
                metadata = json.loads((checkpoint / "adapter_config.json").read_text())
                self.assertEqual(metadata["targets"], list(LORA_TARGETS))
                restored = copy.deepcopy(base)
                inject_lora(restored, rank=2, alpha=2)
                resumed = torch.optim.AdamW(
                    [p for p in restored.parameters() if p.requires_grad]
                )
                load_lora_training_checkpoint(restored, resumed, checkpoint)
                torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)
                exported = Path(directory) / "exported"
                self.assertEqual(export_adapter(checkpoint, exported), 34)
                load_lora_adapter(base, exported)
                torch.testing.assert_close(base(inputs), expected, rtol=0, atol=0)
            finally:
                shutdown_lora_checkpoint_writer()


if __name__ == "__main__":
    unittest.main()
