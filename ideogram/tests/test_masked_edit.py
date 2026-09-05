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
from object_remover.ideogram_edit import IdeogramEditing, CaptionDraftError
from object_remover.matte import refine_alpha

CAPTION = '{"high_level_description":"A clean wall.","compositional_deconstruction":{"background":"A white wall.","elements":[]}}'


class MaskTests(unittest.TestCase):
    def image_mask(self, size=(320, 256)):
        image = Image.new("RGBA", size, (18, 35, 52, 200))
        mask = Image.new("L", size)
        ImageDraw.Draw(mask).rectangle((120, 100, 160, 150), fill=255)
        return png(image), png(mask)

    def test_composite_pixels_and_alpha(self):
        image, mask = self.image_mask()
        region = prepare_region(image, mask, feather=0)
        result = Image.open(io.BytesIO(region.composite(png(Image.new("RGB", region.image.size, "red")))))
        self.assertEqual(result.size, (320, 256))
        self.assertEqual(result.getpixel((0, 0)), (18, 35, 52, 200))
        self.assertEqual(result.getpixel((125, 105)), (255, 0, 0, 200))
        delta = ImageChops.difference(result.convert("RGB"), Image.open(io.BytesIO(image)).convert("RGB"))
        self.assertEqual(delta.getbbox(), (120, 100, 161, 151))

    def test_large_canvas_is_the_whole_model_input(self):
        image, mask = self.image_mask((4096, 4096))
        region = prepare_region(image, mask, feather=8)
        self.assertEqual(region.original.size, (4096, 4096))
        self.assertEqual(region.image.size, (4096, 4096))
        self.assertEqual(region.image.tobytes(), Image.open(io.BytesIO(image)).convert("RGB").tobytes())
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
            prepare_region(image, mask, feather=-1)

    def test_narrow_and_tall_sources_are_only_alignment_trimmed(self):
        for size in [(2048, 37), (37, 2048), (1237, 873)]:
            source = Image.new("RGB", size, "blue")
            mask = Image.new("L", size, 255)
            region = prepare_region(png(source), png(mask), feather=0)
            expected = source.crop((0, 0, size[0] // 16 * 16, size[1] // 16 * 16))
            x1, y1, x2, y2 = region.content_box
            self.assertEqual((x2-x1, y2-y1), expected.size)
            self.assertEqual(region.mask.getbbox(), region.content_box)
            self.assertEqual(region.image.tobytes(), expected.tobytes())
            self.assertEqual(Image.open(io.BytesIO(region.composite(png(region.image)))).tobytes(), expected.tobytes())
            self.assertEqual(region.review()["trimmed_pixels"], [size[0] % 16, size[1] % 16])
            self.assertEqual(region.image.width % 16, 0)
            self.assertEqual(region.image.height % 16, 0)

    def test_large_rectangular_review_does_not_run_models(self):
        image = png(Image.new("RGB", (4096, 2400), "gray"))
        mask = png(Image.new("L", (4096, 2400), 255))
        region = prepare_region(image, mask, feather=0)
        info = region.review()
        self.assertEqual(info["processing_size"], (4096, 2400))
        self.assertEqual(info["source_size"], (4096, 2400))
        self.assertFalse(info["downscaled"])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            region.composite(png(Image.new("RGB", (256, 256))))

    def test_alpha_refinement_is_tile_consistent_and_preserves_dark_foreground(self):
        source = Image.new("RGB", (301, 179), "white")
        ImageDraw.Draw(source).rectangle((65, 30, 180, 160), fill="black")
        alpha = Image.new("L", source.size)
        ImageDraw.Draw(alpha).rectangle((65, 30, 180, 160), fill=255)
        full = refine_alpha(source, alpha, tile_size=1024)
        tiled = refine_alpha(source, alpha, tile_size=64)
        self.assertIsNone(ImageChops.difference(full, tiled).getbbox())
        result = Image.open(io.BytesIO(cutout(png(source), png(tiled))))
        self.assertEqual(result.getpixel((100, 100)), (0, 0, 0, 255))
        self.assertEqual(result.getpixel((0, 0)), (255, 255, 255, 0))


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
        seeded = caption_graph("i.png", "local.safetensors", "Remove chair", "Schema", seed=2**64-1)
        self.assertEqual(seeded["3"]["inputs"]["sampling_mode"], "on")
        self.assertEqual(seeded["3"]["inputs"]["sampling_mode.seed"], 2**64-1)
        self.assertEqual(graph["3"]["inputs"]["sampling_mode"], "off")

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
        with self.assertRaisesRegex(CaptionDraftError, "after one retry") as error:
            edit.caption(region, "Remove the object")
        self.assertEqual(error.exception.draft, "bad")
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
        edit.engine.image_output.return_value = png(Image.new("RGB", (320, 256), "red"))
        edit.caption = Mock(side_effect=AssertionError("Must not generate a supplied caption"))
        image, mask = MaskTests().image_mask()
        result = edit.edit(image, mask, instruction="", caption=CAPTION,
                           feather=0, invert=False, steps=4, seed=1, guidance=4, strength=1)
        self.assertEqual(Image.open(io.BytesIO(result)).size, (320, 256))
        edit.engine.stop.assert_called_once()
        edit.engine.reset_mock()
        edit.engine.run.side_effect = RuntimeError("GPU OOM")
        with self.assertRaisesRegex(RuntimeError, "GPU OOM"):
            edit.edit(image, mask, instruction="", caption=CAPTION,
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
            models.foreground_mask.return_value = mask
            models.caption_ideogram.return_value = CAPTION
            plan = client.post("/api/ideogram/prepare", files=files)
            self.assertEqual(plan.status_code, 200, plan.text)
            self.assertEqual(plan.json()["source_size"], [320, 256])
            models.edit_ideogram.assert_not_called()
            models.caption_ideogram.assert_not_called()
            self.assertEqual(client.post("/api/ideogram/caption", files=files, data={
                "instruction": "Remove chair", "caption_seed": str(2**64-1)}).status_code, 200)
            self.assertEqual(models.caption_ideogram.call_args.kwargs["caption_seed"], 2**64-1)
            models.caption_ideogram.side_effect = CaptionDraftError("bad JSON", "unfinished draft")
            rejected = client.post("/api/ideogram/caption", files=files, data={"instruction": "Remove chair"})
            self.assertEqual(rejected.status_code, 422)
            self.assertEqual(rejected.json()["draft"], "unfinished draft")
            models.caption_ideogram.side_effect = None
            self.assertEqual(client.post("/api/foreground-mask", files={"image": files["image"]}).status_code, 200)
            self.assertEqual(client.post("/api/remove", files=files, data={"engine": "ideogram", "reviewed": "true"}).status_code, 400)
            models.edit_ideogram.assert_not_called()
            response = client.post("/api/remove", files=files, data={"engine": "ideogram", "caption": CAPTION,
                "resolution": "1536", "padding": "90", "mask_feather": "12", "invert": "false", "strength": ".8"})
            self.assertEqual(response.status_code, 200, response.text)
            options = models.edit_ideogram.call_args.kwargs
            self.assertFalse(options["invert"])
            self.assertNotIn("resolution", options)
            self.assertNotIn("padding", options)
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
