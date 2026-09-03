from __future__ import annotations

import io
import unittest
from types import SimpleNamespace

from server import ApiHandler, _decode_image_array, _vision_prompt


ONE_PIXEL_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class VisionPromptTests(unittest.TestCase):
    def test_explain_prompt_is_grounded(self) -> None:
        prompt = _vision_prompt("explain", "ignored", "auto", "Spanish")
        self.assertIn("Do not invent", prompt)
        self.assertIn("Respond in Spanish", prompt)

    def test_translate_prompt_uses_language_hint_and_target(self) -> None:
        prompt = _vision_prompt("translate", "", "zh", "Spanish")
        self.assertIn("Chinese", prompt)
        self.assertIn("Spanish", prompt)
        self.assertIn("Return only the translation", prompt)

    def test_custom_prompt_is_verbatim(self) -> None:
        self.assertEqual(
            _vision_prompt("custom", "  Count every label.  ", "auto", "English"),
            "Count every label.",
        )

    def test_custom_prompt_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "prompt is required"):
            _vision_prompt("custom", "", "auto", "English")

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            _vision_prompt("guess", "", "auto", "English")


class ImageDecodeTests(unittest.TestCase):
    def test_data_url_decodes_to_hwc_rgb(self) -> None:
        image = _decode_image_array(ONE_PIXEL_PNG, 1024, 16)
        self.assertEqual(image.shape, (1, 1, 3))
        self.assertEqual(str(image.dtype), "uint8")

    def test_non_data_url_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "base64 image data URL"):
            _decode_image_array("https://example.test/image.png", 1024, 16)

    def test_decoded_byte_limit_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "decoded image exceeds"):
            _decode_image_array(ONE_PIXEL_PNG, 4, 16)

    def test_invalid_base64_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid base64"):
            _decode_image_array("data:image/png;base64,not-valid!", 1024, 16)

    def test_pillow_decompression_bomb_is_a_client_error(self) -> None:
        from PIL import Image

        previous_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = 0
        try:
            with self.assertRaisesRegex(ValueError, "safe pixel limits"):
                _decode_image_array(ONE_PIXEL_PNG, 1024, 16)
        finally:
            Image.MAX_IMAGE_PIXELS = previous_limit


class RequestBodyTests(unittest.TestCase):
    def test_negative_content_length_is_rejected(self) -> None:
        request = SimpleNamespace(
            headers={"Content-Length": "-1"},
            rfile=io.BytesIO(b"{}"),
            app=SimpleNamespace(max_body_bytes=1024),
        )
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            ApiHandler._body(request)

    def test_route_specific_limit_is_enforced(self) -> None:
        request = SimpleNamespace(
            headers={"Content-Length": "2"},
            rfile=io.BytesIO(b"{}"),
            app=SimpleNamespace(max_body_bytes=1),
        )
        self.assertEqual(ApiHandler._body(request, max_bytes=2), {})


if __name__ == "__main__":
    unittest.main()
