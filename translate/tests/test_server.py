from __future__ import annotations

import io
import unittest
from types import SimpleNamespace

from local_app.server import RequestBodyLimitMiddleware
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


class UiRequestBodyLimitTests(unittest.IsolatedAsyncioTestCase):
    async def _request(
        self,
        *,
        path: str,
        chunks: list[bytes],
        content_length: bytes | None = None,
    ) -> tuple[bool, list[dict[str, object]]]:
        called = False

        async def downstream(scope, receive, send) -> None:
            nonlocal called
            called = True
            while (await receive()).get("more_body", False):
                pass

        messages = [
            {
                "type": "http.request",
                "body": chunk,
                "more_body": index + 1 < len(chunks),
            }
            for index, chunk in enumerate(chunks)
        ]

        async def receive() -> dict[str, object]:
            return messages.pop(0)

        sent: list[dict[str, object]] = []

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        headers = [] if content_length is None else [(b"content-length", content_length)]
        middleware = RequestBodyLimitMiddleware(
            downstream, max_body_bytes=4, max_vision_body_bytes=8
        )
        await middleware(
            {"type": "http", "method": "POST", "path": path, "headers": headers},
            receive,
            send,
        )
        return called, sent

    async def test_chunked_vision_body_is_rejected_before_fastapi(self) -> None:
        called, sent = await self._request(path="/api/vision", chunks=[b"1234", b"56789"])
        self.assertFalse(called)
        self.assertEqual(sent[0]["status"], 413)

    async def test_declared_oversize_is_rejected_without_reading(self) -> None:
        called, sent = await self._request(
            path="/api/translate", chunks=[b"{}"], content_length=b"5"
        )
        self.assertFalse(called)
        self.assertEqual(sent[0]["status"], 413)

    async def test_body_at_limit_replays_to_fastapi(self) -> None:
        called, sent = await self._request(
            path="/api/translate", chunks=[b"12", b"34"]
        )
        self.assertTrue(called)
        self.assertEqual(sent, [])


if __name__ == "__main__":
    unittest.main()
