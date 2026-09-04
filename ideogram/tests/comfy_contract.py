"""Explicit CPU-only checks with the bundled Comfy environment. No weight loading.

cd ideogram && ../minimax/.venv/bin/python tests/comfy_contract.py
"""
import asyncio
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "minimax/runtime")]
sys.argv = [sys.argv[0], "--cpu"]
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import torch
import nodes
from object_remover.comfy_nodes import RowFP8Ops, verify_loaded, layout
from object_remover.ideogram_graph import inpaint_graph, caption_graph
from object_remover.private_comfy import PrivateComfy
from object_remover.regions import png
from PIL import Image


def meta_checkpoint(part, name):
    root = Path(os.getenv("IDEOGRAM4_FP8_MODEL", "~/multimedia/models/ideogram-ai--ideogram-4-fp8")).expanduser()
    with (root / part / name).open("rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    types = {"F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32, "BF16": torch.bfloat16}
    return {key: torch.empty(value["shape"], dtype=types[value["dtype"]], device="meta")
            for key, value in header.items() if key != "__metadata__"}


class ContractTests(unittest.TestCase):
    def test_fp8_math_and_storage(self):
        # Different row scales expose the exact error an ordinary FP8 loader would hide.
        weight = torch.tensor([[1., -2., 3.], [4., 2., -1.]]).to(torch.float8_e4m3fn)
        scale = torch.tensor([.125, 3.], dtype=torch.float32)
        bias = torch.tensor([.5, -.5])
        layer = RowFP8Ops.Linear(3, 2)
        layer.load_state_dict({"weight": weight, "weight_scale": scale, "bias": bias})
        data = torch.tensor([[2., 1., -3.]])
        expected = torch.nn.functional.linear(data, weight.float() * scale[:, None], bias)
        torch.testing.assert_close(layer(data), expected, rtol=0, atol=0)
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.weight_comfy_model_dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.weight.data_ptr(), weight.data_ptr())
        with self.assertRaisesRegex(RuntimeError, "one scale per output row"):
            RowFP8Ops.Linear(3, 2).load_state_dict({"weight": weight, "bias": bias})

    def test_bf16_unquantized_linear(self):
        layer = RowFP8Ops.Linear(3, 2, bias=False)
        layer.load_state_dict({"weight": torch.ones((2, 3), dtype=torch.bfloat16)})
        torch.testing.assert_close(layer(torch.ones((1, 3))), torch.full((1, 2), 3.))

    def test_actual_native_diffusion_checkpoint_layouts_without_loading_weights(self):
        import comfy.model_detection
        for part in ("transformer", "unconditional_transformer"):
            state = meta_checkpoint(part, "diffusion_pytorch_model.safetensors")
            config = comfy.model_detection.model_config_from_unet(state, "")
            config.set_inference_dtype(torch.bfloat16, torch.bfloat16)
            config.custom_operations = RowFP8Ops
            with torch.device("meta"):
                model = config.get_model(state, "", device="meta")
            result = model.diffusion_model.load_state_dict(state, strict=False)
            self.assertEqual(result.missing_keys, [])
            self.assertEqual(result.unexpected_keys, [])
            verify_loaded(layout(state), model.diffusion_model)

    def test_actual_native_text_layout_without_loading_weights(self):
        from comfy.text_encoders.ideogram4 import Qwen3VL8BModel
        state = meta_checkpoint("text_encoder", "model.safetensors")
        state = {"model." + k[len("language_model."):]: v for k, v in state.items()
                 if k.startswith("language_model.") and k != "language_model.norm.weight"}
        with torch.device("meta"):
            model = Qwen3VL8BModel(device="meta", dtype=torch.bfloat16, model_options={"custom_operations": RowFP8Ops})
        result = model.load_sd(state)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        verify_loaded(layout(state), model.transformer)

    def test_comfy_validates_connections_and_values_without_running_graph(self):
        import folder_paths
        import execution
        from object_remover.comfy_nodes import NODE_CLASS_MAPPINGS
        async def check():
            extras = ROOT.parent / "minimax/runtime/comfy_extras"
            for filename in ("nodes_custom_sampler.py", "nodes_differential_diffusion.py", "nodes_ideogram4.py",
                             "nodes_textgen.py", "nodes_preview_any.py", "nodes_mask.py"):
                self.assertTrue(await nodes.load_custom_node(str(extras / filename), module_parent="comfy_extras"))
            nodes.NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS)
            prior = folder_paths.get_input_directory()
            with tempfile.TemporaryDirectory(prefix="ideogram-validation-") as temp:
                folder_paths.set_input_directory(temp)
                try:
                    for name in ("i.png", "m.png"):
                        (Path(temp) / name).write_bytes(png(Image.new("RGB", (256, 256), "white")))
                    caption_model = PrivateComfy().caption_model
                    folder_paths.add_model_folder_path("text_encoders", str(caption_model.parent))
                    graphs = [inpaint_graph("i.png", "m.png", "{}", 512, 512, 20, 42, 4, 1),
                              caption_graph("i.png", caption_model.name, "Remove chair", "Schema"),
                              caption_graph("i.png", caption_model.name, "Remove chair", "Schema", seed=42)]
                    for graph in graphs:
                        result = await execution.validate_prompt("validation-only", graph, None)
                        self.assertTrue(result[0], result)
                finally:
                    folder_paths.set_input_directory(prior)
        asyncio.run(check())

    def test_private_cpu_start_graph_validation_stop(self):
        # Start the actual sidecar in CPU mode on a fresh port/state; don't queue any graph.
        import socket
        with tempfile.TemporaryDirectory(prefix="ideogram-contract-") as temp, socket.socket() as sock:
            sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
            patcher = unittest.mock.patch.dict(os.environ, {"IDEOGRAM_PRIVATE_PORT": str(port),
                    "IDEOGRAM_COMFY_STATE": temp, "IDEOGRAM_COMFY_DEVICE": "cpu",
                    "IDEOGRAM_COMFY_PYTHON": sys.executable})
            patcher.start()
            engine = PrivateComfy()
            try:
                engine.start()
                info = engine.request("GET", "/object_info").json()
                graphs = [inpaint_graph("i.png", "m.png", "{}", 512, 512, 20, 42, 4, 1),
                          caption_graph("i.png", engine.caption_model.name, "Remove chair", "schema")]
                for graph in graphs:
                    for node in graph.values():
                        spec = info[node["class_type"]]
                        supplied = node["inputs"]
                        required = spec["input"].get("required", {})
                        optional = spec["input"].get("optional", {})
                        self.assertFalse(set(required) - set(supplied), (node, required))
                        self.assertFalse(set(supplied) - set(required) - set(optional), (node, spec["input"]))
            except Exception:
                print((Path(temp) / "engine.log").read_text(errors="replace")[-12000:])
                raise
            finally:
                engine.stop()
                patcher.stop()
                self.assertFalse(engine.running)


if __name__ == "__main__":
    from unittest import mock
    # Strip Comfy's CPU switch before unittest parses arguments.
    unittest.main(argv=[sys.argv[0]], verbosity=2)
