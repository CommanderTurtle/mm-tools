"""Explicit CPU-only checks with the bundled Comfy environment. No weight loading.

cd visual-modeling/background-removal/ideogram && ../../../music/minimax/.venv/bin/python tests/comfy_contract.py
Run after the pinned Comfy-Org/Ideogram-4 set is present (see models/which-ones.txt).
"""
import asyncio
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parents[2] / "music/minimax/runtime")]
sys.argv = [sys.argv[0], "--cpu"]
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import nodes
from object_remover.ideogram_graph import (caption_graph, inpaint_graph,
                                           MAIN_MODEL, UNCOND_MODEL, CLIP_MODEL, VAE_MODEL)
from object_remover.private_comfy import PrivateComfy
from object_remover.regions import png
from PIL import Image


class ContractTests(unittest.TestCase):
    def test_pinned_asset_names(self):
        # Graph node inputs must stay aligned with the downloader's pinned file names.
        self.assertEqual(MAIN_MODEL, "ideogram4_int8_convrot.safetensors")
        self.assertEqual(UNCOND_MODEL, "ideogram4_unconditional_nvfp4_mixed.safetensors")
        self.assertEqual(CLIP_MODEL, "qwen3vl_8b_fp8_scaled.safetensors")
        self.assertEqual(VAE_MODEL, "flux2-vae.safetensors")

    def test_comfy_validates_connections_and_values_without_running_graph(self):
        import folder_paths
        import execution
        async def check():
            extras = ROOT.parents[2] / "music/minimax/runtime/comfy_extras"
            for filename in ("nodes_custom_sampler.py", "nodes_differential_diffusion.py", "nodes_ideogram4.py",
                             "nodes_textgen.py", "nodes_preview_any.py", "nodes_mask.py"):
                self.assertTrue(await nodes.load_custom_node(str(extras / filename), module_parent="comfy_extras"))
            prior = folder_paths.get_input_directory()
            with tempfile.TemporaryDirectory(prefix="ideogram-validation-") as temp:
                folder_paths.set_input_directory(temp)
                try:
                    for name in ("i.png", "m.png"):
                        (Path(temp) / name).write_bytes(png(Image.new("RGB", (256, 256), "white")))
                    model = PrivateComfy().model
                    for folder in ("diffusion_models", "text_encoders", "vae"):
                        folder_paths.add_model_folder_path(folder, str(model / folder))
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
    # Strip Comfy's CPU switch before unittest parses arguments.
    unittest.main(argv=[sys.argv[0]], verbosity=2)
