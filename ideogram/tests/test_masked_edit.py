"""CPU-only regression checks; no weights, network, or model inference.

cd ideogram && .venv/bin/python -m unittest discover -s tests
"""
import io
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from PIL import Image, ImageDraw, ImageChops
from object_remover.regions import prepare_region, png, cutout
from object_remover.ideogram_graph import inpaint_graph, caption_graph
from object_remover.private_comfy import PrivateComfy, caption_json
from object_remover.ideogram_edit import IdeogramEditing

CAPTION = '{"high_level_description":"A clean wall.","compositional_deconstruction":{"background":"A white wall.","elements":[]}}'


class MaskTests(unittest.TestCase):
    def image_mask(self, size=(320, 256)):
        image = Image.new("RGBA", size, (18, 35, 52, 200))
        mask = Image.new("L", size)
        ImageDraw.Draw(mask).rectangle((120, 100, 160, 150), fill=255)
        return png(image), png(mask)

    def test_composite_pixels_and_alpha(self):
        image, mask = self.image_mask()
        region = prepare_region(image, mask, padding=16, feather=0)
        result = Image.open(io.BytesIO(region.composite(png(Image.new("RGB", region.image.size, "red")))))
        self.assertEqual(result.size, (320, 256))
        self.assertEqual(result.getpixel((0, 0)), (18, 35, 52, 200))
        self.assertEqual(result.getpixel((125, 105)), (255, 0, 0, 200))
        delta = ImageChops.difference(result.convert("RGB"), Image.open(io.BytesIO(image)).convert("RGB"))
        self.assertEqual(delta.getbbox(), (120, 100, 161, 151))

    def test_large_canvas_crop_not_whole_image_resize(self):
        image, mask = self.image_mask((4096, 4096))
        region = prepare_region(image, mask, resolution=1024, padding=128, feather=8)
        self.assertEqual(region.original.size, (4096, 4096))
        self.assertLess(region.image.width, 1024)
        self.assertEqual(region.image.width % 16, 0)

    def test_invert_keeps_foreground(self):
        image, mask = self.image_mask()
        region = prepare_region(image, mask, invert=True, feather=0)
        result = Image.open(io.BytesIO(region.composite(png(Image.new("RGB", region.image.size, "red")))))
        self.assertEqual(result.getpixel((125, 105)), (18, 35, 52, 200))
        self.assertEqual(result.getpixel((0, 0)), (255, 0, 0, 200))

    def test_cutout_no_resizing(self):
        image, mask = self.image_mask()
        result = Image.open(io.BytesIO(cutout(image, mask)))
        self.assertEqual(result.size, (320, 256))
        self.assertEqual(result.getpixel((125, 105)), (18, 35, 52, 200))
        self.assertEqual(result.getpixel((0, 0)), (18, 35, 52, 0))

    def test_mask_validation(self):
        image, mask = self.image_mask()
        for bad in [png(Image.new("L", (10, 10))), png(Image.new("L", (320, 256)))]:
            with self.assertRaises(ValueError):
                prepare_region(image, bad)
        with self.assertRaises(ValueError):
            prepare_region(image, mask, resolution=123)


class GraphTests(unittest.TestCase):
    def test_graph_contract(self):
        graph = inpaint_graph("i.png", "m.png", CAPTION, 1024, 768, 20, 42, 4, 1)
        self.assertEqual(graph["17"]["inputs"]["sigmas"], ["14", 1])
        self.assertEqual(graph["12"]["inputs"]["model_negative"], ["5", 0])
        self.assertEqual(graph["11"]["class_type"], "DifferentialDiffusion")
        self.assertEqual(graph["10"]["class_type"], "SetLatentNoiseMask")
        self.assertEqual(graph["18"]["class_type"], "VAEDecodeTiled")
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    self.assertIn(value[0], graph)

    def test_caption_is_local_vision(self):
        graph = caption_graph("i.png", "local.safetensors", "Remove chair", "Schema")
        self.assertEqual(graph["3"]["inputs"]["image"], ["1", 0])
        self.assertFalse(graph["3"]["inputs"]["thinking"])
        self.assertIn("Remove chair", graph["3"]["inputs"]["prompt"])

    def test_caption_validation(self):
        self.assertEqual(json.loads(caption_json("```json\n" + CAPTION + "\n```")), json.loads(CAPTION))
        obj = json.loads(CAPTION)
        obj["aspect_ratio"] = "4:3"
        self.assertNotIn("aspect_ratio", json.loads(caption_json(json.dumps(obj))))
        for bad in ["plain text", "[]", "{}", '{"high_level_description":4}']:
            with self.assertRaises(ValueError):
                caption_json(bad)


class LifecycleTests(unittest.TestCase):
    def test_caption_repair_is_bounded_and_keeps_clean_image(self):
        edit = IdeogramEditing()
        edit.engine = Mock()
        edit.engine.caption_model.name = "caption.safetensors"
        edit.engine.run.side_effect = [{"4": {"text": ["{bad}"]}}, {"4": {"text": [CAPTION]}}]
        image, mask = MaskTests().image_mask()
        region = prepare_region(image, mask)
        self.assertEqual(json.loads(edit.caption(region, "Remove the object")), json.loads(CAPTION))
        self.assertEqual(edit.engine.run.call_count, 2)
        uploaded = next(iter(edit.engine.run.call_args.args[1].values()))
        self.assertEqual(uploaded, png(region.image))
        edit.engine.stop.assert_called_once()
        edit.engine.reset_mock()
        edit.engine.run.side_effect = [{"4": {"text": ["bad"]}}, {"4": {"text": ["bad"]}}]
        with self.assertRaisesRegex(ValueError, "after one retry"):
            edit.caption(region, "Remove the object")
        self.assertEqual(edit.engine.run.call_count, 2)
        edit.engine.stop.assert_called_once()

    def test_occupied_port_never_reused(self):
        engine = PrivateComfy()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0)); sock.listen()
            engine.port = sock.getsockname()[1]
            with patch.object(engine, "status", return_value={"missing": []}):
                with self.assertRaisesRegex(RuntimeError, "occupied"):
                    engine.start()
        self.assertIsNone(engine.process)

    def test_no_external_proxy(self):
        self.assertFalse(PrivateComfy().http.trust_env)

    def test_output_path_guard(self):
        engine = PrivateComfy()
        with tempfile.TemporaryDirectory() as temp:
            engine.state = Path(temp)
            (engine.state / "output").mkdir()
            with self.assertRaisesRegex(RuntimeError, "Invalid private output path"):
                engine.image_output({"19": {"images": [{"filename": "../../secrets", "type": "output"}]}})

    def test_manual_caption_skips_magic_and_always_stops(self):
        edit = IdeogramEditing()
        edit.engine = Mock()
        edit.engine.image_output.return_value = png(Image.new("RGB", (256, 256), "red"))
        edit.caption = Mock(side_effect=AssertionError("Must not generate a supplied caption"))
        image, mask = MaskTests().image_mask()
        result = edit.edit(image, mask, instruction="", caption=CAPTION, resolution=512, padding=16,
                           feather=0, invert=False, steps=4, seed=1, guidance=4, strength=1)
        self.assertEqual(Image.open(io.BytesIO(result)).size, (320, 256))
        edit.engine.stop.assert_called_once()
        edit.engine.reset_mock()
        edit.engine.run.side_effect = RuntimeError("GPU OOM")
        with self.assertRaisesRegex(RuntimeError, "GPU OOM"):
            edit.edit(image, mask, instruction="", caption=CAPTION, resolution=512, padding=16,
                      feather=0, invert=False, steps=4, seed=1, guidance=4, strength=1)
        edit.engine.stop.assert_called_once()


class APITests(unittest.TestCase):
    def test_existing_routes_and_new_ideogram_form(self):
        try:
            from fastapi.testclient import TestClient
        except (ImportError, RuntimeError):
            self.skipTest("API client checks require the optional httpx test dependency")
        from object_remover.server import app
        image, mask = MaskTests().image_mask()
        files = {"image": ("image.png", image, "image/png"), "mask": ("mask.png", mask, "image/png")}
        with patch("object_remover.server.models") as models, TestClient(app) as client:
            models.edit_ideogram.return_value = image
            models.remove_object.return_value = image
            models.remove_background.return_value = image
            response = client.post("/api/remove", files=files, data={"engine": "ideogram", "caption": CAPTION,
                "resolution": "1536", "padding": "90", "mask_feather": "12", "invert": "false", "strength": ".8"})
            self.assertEqual(response.status_code, 200, response.text)
            options = models.edit_ideogram.call_args.kwargs
            self.assertFalse(options["invert"])
            self.assertEqual(options["resolution"], 1536)
            self.assertEqual(options["feather"], 12)
            self.assertEqual(client.post("/api/remove", files=files, data={"engine": "objectclear"}).status_code, 200)
            models.remove_object.assert_called_once()
            self.assertEqual(client.post("/api/background", files={"image": files["image"]}).status_code, 200)
            self.assertEqual(client.post("/api/cutout", files=files).status_code, 200)
            self.assertEqual(client.post("/api/remove", files=files, data={"engine": "not-an-engine"}).status_code, 400)
            models.edit_ideogram.side_effect = ValueError("bad caption")
            self.assertEqual(client.post("/api/remove", files=files, data={"engine": "ideogram"}).status_code, 400)
            models.edit_ideogram.side_effect = RuntimeError("engine error")
            self.assertEqual(client.post("/api/remove", files=files, data={"engine": "ideogram"}).status_code, 500)


if __name__ == "__main__":
    unittest.main()
