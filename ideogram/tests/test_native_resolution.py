"""Regression: model wrappers receive source pixels, never a resized selection."""
import io
import unittest
from unittest.mock import Mock
from types import SimpleNamespace

import numpy as np
from PIL import Image

from object_remover.native import align_source, read_mask
from object_remover.regions import png
from object_remover.model_backend import EditingModels


class NativeResolutionTests(unittest.TestCase):
    def test_alignment_preserves_every_surviving_pixel(self):
        for width, height in [(2400, 1600), (2413, 1619), (2160, 1440), (801, 607)]:
            with self.subTest(size=(width, height)):
                pixels = np.random.default_rng(42).integers(0, 256, (height, width, 3), dtype=np.uint8)
                source = Image.fromarray(pixels)
                for grid in (16, 32):
                    result = align_source(source, grid)
                    self.assertEqual(result.size, (width // grid * grid, height // grid * grid))
                    self.assertTrue(np.array_equal(np.asarray(result), pixels[:result.height, :result.width]))
                    self.assertLess(width - result.width, grid)
                    self.assertLess(height - result.height, grid)

    def test_objectclear_receives_whole_source_not_mask_crop(self):
        for size in [(2400, 1600), (2413, 1619)]:
            pixels = np.random.default_rng(5).integers(0, 256, (*size[::-1], 3), dtype=np.uint8)
            image = Image.fromarray(pixels)
            mask = Image.new("L", size)
            mask.paste(255, (120, 100, 160, 150))
            models = EditingModels()
            models._load_objectclear = Mock()
            models._torch = Mock(return_value=Mock())
            expected = align_source(image, 16)
            models._objectclear = Mock(return_value=SimpleNamespace(images=[expected.copy()]))
            output = models.remove_object(png(image), png(mask), steps=20, guidance=2.5, seed=42)
            sent = models._objectclear.call_args.kwargs
            self.assertEqual(sent["image"].size, expected.size)
            self.assertEqual(sent["image"].tobytes(), expected.tobytes())
            self.assertEqual(sent["mask_image"].getbbox(), (120, 100, 160, 150))
            self.assertEqual(sent["height"], expected.height)
            self.assertEqual(sent["width"], expected.width)
            self.assertEqual(Image.open(io.BytesIO(output)).tobytes(), expected.tobytes())
            models._objectclear.return_value.images = [Image.new("RGB", (512, 512))]
            with self.assertRaisesRegex(ValueError, "refusing to resize"):
                models.remove_object(png(image), png(mask), steps=20, guidance=2.5, seed=42)

    def test_mismatched_masks_and_too_small_images_are_not_scaled(self):
        from object_remover.core import prepare_mask
        with self.assertRaisesRegex(ValueError, "not resized"):
            prepare_mask(np.ones((512, 512), dtype=np.uint8), (1600, 2400), grow=0)
        with self.assertRaisesRegex(ValueError, "not resized"):
            read_mask(png(Image.new("L", (512, 512), 255)), (2400, 1600))
        with self.assertRaisesRegex(ValueError, "upscaling is disabled"):
            align_source(Image.new("RGB", (15, 15)), 16)

    def test_birefnet_receives_native_tensor_and_keeps_source_rgb(self):
        import torch
        for width, height in [(2400, 1600), (2413, 1619)]:
            pixels = np.random.default_rng(6).integers(0, 256, (height, width, 4), dtype=np.uint8)
            image = Image.fromarray(pixels)
            expected = align_source(image, 32)
            models = EditingModels()
            models._load_birefnet = Mock()
            models._device = "cpu"
            models._birefnet = Mock()
            models._birefnet.parameters.side_effect = lambda: iter([torch.zeros(1)])
            prediction = torch.zeros((1, 1, expected.height, expected.width))
            models._birefnet.return_value = [prediction]
            output = Image.open(io.BytesIO(models.remove_background(png(image))))
            tensor = models._birefnet.call_args.args[0]
            self.assertEqual(tuple(tensor.shape), (1, 3, expected.height, expected.width))
            # Undo only the existing normalization: every model-input RGB pixel is retained.
            mean = torch.tensor([.485, .456, .406])[:, None, None]
            std = torch.tensor([.229, .224, .225])[:, None, None]
            restored = ((tensor[0] * std + mean) * 255).round().byte().permute(1, 2, 0).numpy()
            self.assertTrue(np.array_equal(restored, np.asarray(expected.convert("RGB"))))
            self.assertEqual(output.size, expected.size)
            self.assertEqual(output.convert("RGB").tobytes(), expected.convert("RGB").tobytes())
            self.assertTrue(np.all(np.asarray(output.getchannel("A")) <= np.asarray(expected.getchannel("A"))))
            models._birefnet.return_value = [torch.zeros((1, 1, 1024, 1024))]
            with self.assertRaisesRegex(ValueError, "refusing to upscale"):
                models.remove_background(png(image))

    def test_objectclear_memory_error_never_retries_with_a_smaller_image(self):
        models = EditingModels()
        models._load_objectclear = Mock()
        models._torch = Mock(return_value=Mock())
        models._objectclear = Mock(side_effect=RuntimeError("CUDA out of memory"))
        image = Image.new("RGB", (2400, 1600))
        mask = Image.new("L", image.size, 255)
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            models.remove_object(png(image), png(mask), steps=20, guidance=2.5, seed=42)
        models._objectclear.assert_called_once()
        self.assertEqual(models._objectclear.call_args.kwargs["image"].size, image.size)


if __name__ == "__main__":
    unittest.main()
