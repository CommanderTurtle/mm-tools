import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

spec = importlib.util.spec_from_file_location(
    "jev_httpd", Path(__file__).parents[1] / "server.py"
)
httpd = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = httpd
spec.loader.exec_module(httpd)


@contextmanager
def running(server):
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class FakeNInfer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.respond(None)

    def do_POST(self):
        self.respond(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

    def respond(self, body):
        self.server.calls.append((self.path, body, self.headers.get("Authorization")))
        status = 200
        if self.path == "/health":
            value = {"status": "ok"}
        elif self.path == "/v1/models":
            value = {"object": "list", "data": self.server.models}
        elif self.path == "/v1/systemone":
            criteria = body["questions"]["language"]["criteria"]
            probabilities = {code: 0.01 / (len(criteria) - 1) for code in criteria}
            probabilities["de"] = 0.99
            value = {
                "model": "resident",
                "answers": {
                    "language": {
                        "type": "choice",
                        "choice": "de",
                        "confidence": 0.98,
                        "probabilities": probabilities,
                    }
                },
            }
        elif self.path == "/v1/chat/completions":
            if self.server.delay:
                time.sleep(self.server.delay)
            if "tools" in body:
                text = json.loads(body["messages"][1]["content"])["text"]
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "submit_translation",
                                "arguments": json.dumps(
                                    {"translation": self.server.translate(text)}
                                ),
                            },
                        }
                    ],
                }
                finish = "tool_calls"
            else:
                message, finish = (
                    {"role": "assistant", "content": "A blue square"},
                    "stop",
                )
            value = {
                "model": "resident",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            }
        else:
            status, value = 404, {"error": "not found"}
        if self.server.override:
            status, value = self.server.override(self.path, body, status, value)
        raw = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        if getattr(self.server, "redirect", None):
            self.send_header("Location", self.server.redirect)
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeNInfer)
        self.upstream.daemon_threads = True
        self.upstream.calls = []
        self.upstream.models = [
            {"id": "resident", "object": "model", "max_model_len": 8192}
        ]
        self.upstream.override = None
        self.upstream.delay = 0
        self.upstream.translate = lambda text: {
            "Guten Morgen": "Good morning",
            "Hallo": "Hello",
        }.get(text, text)
        self.origin = self.enterContext(running(self.upstream))
        self.adapter = httpd.Adapter(
            httpd.Config(endpoint=self.origin, upstream_key="upstream-secret")
        )
        self.api = self.enterContext(
            running(httpd.Server(("127.0.0.1", 0), self.adapter))
        )

    def call(self, path, body=None, key="", raw=None):
        data = (
            raw
            if raw is not None
            else None
            if body is None
            else json.dumps(body).encode()
        )
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = Request(self.api + path, data=data, headers=headers)
        try:
            with urlopen(request, timeout=3) as response:
                return response.status, json.load(response), response.headers
        except HTTPError as exc:
            return exc.code, json.load(exc), exc.headers

    def test_startup_and_idle_do_not_call_endpoint(self):
        time.sleep(0.02)
        self.assertEqual(self.upstream.calls, [])

    def test_health_only_gets_metadata_and_never_loads_model(self):
        status, result, _ = self.call("/health")
        self.assertEqual(status, 200)
        self.assertTrue(result["loaded"])
        self.assertFalse(result["cloud"])
        self.assertFalse(result["owns_model"])
        self.assertEqual(result["model"], "resident")
        self.assertEqual(
            [row[0] for row in self.upstream.calls], ["/health", "/v1/models"]
        )
        self.assertTrue(
            all(row[2] == "Bearer upstream-secret" for row in self.upstream.calls)
        )

    def test_known_source_goes_straight_to_v1_tool_call(self):
        status, result, _ = self.call(
            "/translate",
            {
                "text": " \nGuten Morgen\t",
                "source_language": "German",
                "target_language": "English",
            },
        )
        self.assertEqual((status, result["translation"]), (200, " \nGood morning\t"))
        self.assertEqual(result["source_language"], "de")
        self.assertEqual(
            [row[0] for row in self.upstream.calls],
            ["/v1/models", "/v1/chat/completions"],
        )
        body = self.upstream.calls[-1][1]
        self.assertFalse(body["stream"])
        self.assertFalse(body["parallel_tool_calls"])
        self.assertEqual(body["tools"][0]["function"]["name"], "submit_translation")
        self.assertEqual(body["tool_choice"]["function"]["name"], "submit_translation")
        self.assertEqual(len(body["messages"]), 2)

    def test_auto_language_makes_one_systemone_decision(self):
        status, result, _ = self.call(
            "/v1/translate",
            {"text": "Guten Morgen", "source_language": "Detect", "route": "inbound"},
        )
        self.assertEqual((status, result["translation"]), (200, "Good morning"))
        self.assertEqual(result["route"], "inbound")
        self.assertEqual(result["source_confidence"], 0.99)
        self.assertEqual(
            [row[0] for row in self.upstream.calls],
            ["/v1/models", "/v1/systemone", "/v1/chat/completions"],
        )

    def test_detect_route_matches_vox(self):
        status, result, _ = self.call("/v1/detect", {"text": "Guten Morgen"})
        self.assertEqual(
            (status, result["language"], result["confidence"]), (200, "de", 0.99)
        )

    def test_no_linguistic_input_does_not_infer(self):
        status, result, _ = self.call("/translate", {"text": " 1234 \n"})
        self.assertEqual(
            (status, result["translation"], result["source_language"]),
            (200, " 1234 \n", "und"),
        )
        self.assertEqual([row[0] for row in self.upstream.calls], ["/v1/models"])

    def test_same_language_does_not_infer(self):
        status, result, _ = self.call(
            "/translate",
            {"text": "Hello", "source_language": "en", "target_language": "English"},
        )
        self.assertEqual((status, result["translation"]), (200, "Hello"))
        self.assertEqual(len(self.upstream.calls), 1)

    def test_long_unicode_chunks_reassemble_exactly(self):
        self.adapter = httpd.Adapter(httpd.Config(endpoint=self.origin, chunk_bytes=80))
        text = "  Hallo 世界🌎\n" * 30
        result = self.adapter.translate(
            {"text": text}, self.upstream.models[0], time.monotonic() + 3
        )
        self.assertEqual(result["translation"], text)
        self.assertEqual(
            sum(row[0] == "/v1/systemone" for row in self.upstream.calls), 1
        )
        chunks = [
            json.loads(row[1]["messages"][1]["content"])["text"]
            for row in self.upstream.calls
            if row[0] == "/v1/chat/completions"
        ]
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(part.encode()) <= 80 for part in chunks))

    def test_legacy_chat_completion_prompt_contract(self):
        status, result, _ = self.call(
            "/v1/chat/completions",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "The source language is German. The target language is English. <source_text>Guten Morgen</source_text>",
                    }
                ]
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(result["choices"][0]["message"]["content"], "Good morning")
        self.assertEqual(result["model"], "resident")

    def test_vision_uses_existing_multimodal_model_without_jev(self):
        data_url = "data:image/png;base64,iVBORw0KGgo="
        status, result, _ = self.call(
            "/v1/vision", {"image_data_url": data_url, "mode": "explain"}
        )
        self.assertEqual((status, result["output"]), (200, "A blue square"))
        body = self.upstream.calls[-1][1]
        self.assertEqual(
            body["messages"][0]["content"][1]["image_url"]["url"], data_url
        )
        self.assertNotIn("tools", body)
        self.assertFalse(any(row[0] == "/v1/systemone" for row in self.upstream.calls))

    def test_remote_vision_urls_are_not_fetched(self):
        status, result, _ = self.call(
            "/vision", {"image_data_url": "https://example.test/private.png"}
        )
        self.assertEqual(status, 400)
        self.assertIn("never a remote URL", result["error"])
        self.assertEqual(len(self.upstream.calls), 1)

    def test_unload_load_only_pause_adapter(self):
        self.assertEqual(self.call("/unload", {})[0], 200)
        self.assertFalse(self.call("/health")[1]["loaded"])
        self.assertEqual(self.call("/translate", {"text": "Hallo"})[0], 503)
        self.assertEqual(self.call("/load", {})[0], 200)
        self.assertFalse(
            any(row[0] in ("/load", "/unload") for row in self.upstream.calls)
        )

    def test_malformed_tool_output_is_repaired_once_without_decisions(self):
        count = 0

        def override(path, body, status, value):
            nonlocal count
            if path == "/v1/chat/completions":
                count += 1
                if count == 1:
                    value["choices"][0]["message"] = {"content": "untrusted prose"}
            return status, value

        self.upstream.override = override
        status, result, _ = self.call(
            "/translate", {"text": "Hallo", "source_language": "de"}
        )
        self.assertEqual((status, result["translation"], count), (200, "Hello", 2))
        self.assertFalse(any(row[0] == "/v1/systemone" for row in self.upstream.calls))

    def test_unsolicited_tool_cannot_execute_and_fails_after_one_repair(self):
        def override(path, body, status, value):
            if path == "/v1/chat/completions":
                value["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = (
                    "run_shell"
                )
            return status, value

        self.upstream.override = override
        status, result, _ = self.call(
            "/translate", {"text": "Hallo", "source_language": "de"}
        )
        self.assertEqual(status, 502)
        self.assertNotIn("translation", result)
        self.assertEqual(
            sum(row[0] == "/v1/chat/completions" for row in self.upstream.calls), 2
        )

    def test_truncation_never_returns_partial_translation(self):
        def override(path, body, status, value):
            if path == "/v1/chat/completions":
                value["choices"][0]["finish_reason"] = "length"
            return status, value

        self.upstream.override = override
        status, result, _ = self.call(
            "/translate", {"text": "Hallo", "source_language": "de"}
        )
        self.assertEqual(status, 502)
        self.assertNotIn("translation", result)

    def test_incomplete_language_distribution_is_not_silently_trusted(self):
        def override(path, body, status, value):
            if path == "/v1/systemone":
                value["answers"]["language"]["probabilities"] = {"de": 1}
            return status, value

        self.upstream.override = override
        self.assertEqual(self.call("/detect", {"text": "Hallo"})[0], 502)

    def test_malformed_language_answer_is_an_upstream_error(self):
        for answer in (None, [], "German"):
            self.upstream.override = lambda path, body, status, value, answer=answer: (
                (200, {"answers": {"language": answer}})
                if path == "/v1/systemone"
                else (status, value)
            )
            with self.subTest(answer=answer):
                self.assertEqual(self.call("/detect", {"text": "Hallo"})[0], 502)

    def test_bounded_queue_reports_backpressure(self):
        for _ in range(17):
            self.assertTrue(self.adapter.pending.acquire(blocking=False))
        try:
            status, _, headers = self.call("/translate", {"text": "Hallo"})
            self.assertEqual(status, 429)
            self.assertEqual(headers["Retry-After"], "1")
            self.assertEqual(self.upstream.calls, [])
        finally:
            for _ in range(17):
                self.adapter.pending.release()

    def test_body_limits_are_route_specific(self):
        adapter = httpd.Adapter(httpd.Config(endpoint=self.origin, max_body_bytes=32))
        self.api = self.enterContext(running(httpd.Server(("127.0.0.1", 0), adapter)))
        self.assertEqual(self.call("/translate", {"text": "a" * 100})[0], 413)
        self.assertEqual(self.upstream.calls, [])
        self.assertEqual(
            self.call(
                "/vision",
                {
                    "image_data_url": "data:image/png;base64,iVBORw0KGgo=",
                    "mode": "explain",
                },
            )[0],
            200,
        )

    def test_upstream_redirect_does_not_forward_bearer(self):
        other = ThreadingHTTPServer(("127.0.0.1", 0), FakeNInfer)
        other.calls = []
        self.upstream.redirect = self.enterContext(running(other)) + "/secret"
        self.upstream.override = lambda path, body, status, value: (302, {})
        self.assertEqual(self.call("/health")[0], 503)
        self.assertEqual(other.calls, [])

    def test_jev_missing_route_is_actionable(self):
        self.upstream.override = lambda path, body, status, value: (
            (404, {}) if path == "/v1/systemone" else (status, value)
        )
        status, result, _ = self.call("/detect", {"text": "Hallo"})
        self.assertEqual(status, 502)
        self.assertIn("--jev", result["error"])

    def test_upstream_backpressure_is_propagated_without_retry_storm(self):
        self.upstream.override = lambda path, body, status, value: (
            (429, {"error": "secret"})
            if path == "/v1/chat/completions"
            else (status, value)
        )
        status, result, headers = self.call(
            "/translate", {"text": "Hallo", "source_language": "de"}
        )
        self.assertEqual(status, 429)
        self.assertEqual(headers["Retry-After"], "1")
        self.assertNotIn("secret", result["error"])
        self.assertEqual(
            sum(row[0] == "/v1/chat/completions" for row in self.upstream.calls), 1
        )

    def test_mv3_parallel_fragments_are_serialized_not_dropped(self):
        self.upstream.delay = 0.03
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(
                pool.map(
                    lambda _: self.call(
                        "/translate", {"text": "Hallo", "source_language": "de"}
                    ),
                    range(4),
                )
            )
        self.assertTrue(all(result[0] == 200 for result in results))

    @unittest.skipUnless(
        os.environ.get("VOX_MV3_DIR") and shutil.which("bun"),
        "optional: set VOX_MV3_DIR to a Vox mv3-extension checkout and install Bun",
    )
    def test_real_vox_mv3_translator_against_adapter(self):
        script = """
import {pathToFileURL} from 'node:url';
import {join} from 'node:path';
import assert from 'node:assert/strict';
const {translateWithVox} = await import(pathToFileURL(join(process.env.VOX_MV3_DIR, 'translator.js')).href);
const post = async (path, body) => {
  const response = await fetch(process.env.TRANSLATOR_TEST_URL + path, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const result = await response.json();
  assert.equal(response.status, 200, JSON.stringify(result));
  return result;
};
for (const sourceLanguage of ['German', 'Detect']) {
  const result = await translateWithVox({text:' \\nGuten Morgen\\t',sourceLanguage,targetLanguage:'English',chunkCharacters:400,hintLanguage:'German',post});
  assert.equal(result, ' \\nGood morning\\t');
}
console.log('Actual Vox MV3 translator: known source and Detect passed');
"""
        result = subprocess.run(
            [shutil.which("bun"), "--eval", script],
            env={**os.environ, "TRANSLATOR_TEST_URL": self.api},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_timeout_releases_admission(self):
        self.upstream.delay = 0.1
        adapter = httpd.Adapter(httpd.Config(endpoint=self.origin, timeout=0.03))
        with (
            self.assertRaises(httpd.ServiceError) as error,
            adapter.admitted() as deadline,
        ):
            adapter.translate(
                {"text": "Hallo", "source_language": "de"},
                self.upstream.models[0],
                deadline,
            )
        self.assertEqual(error.exception.status, 504)
        self.assertTrue(adapter.slots.acquire(blocking=False))
        adapter.slots.release()

    def test_authentication_stops_before_any_upstream_request(self):
        protected = httpd.Adapter(
            httpd.Config(endpoint=self.origin, api_key="client-secret")
        )
        self.api = self.enterContext(running(httpd.Server(("127.0.0.1", 0), protected)))
        self.assertEqual(self.call("/health")[0], 401)
        self.assertEqual(self.upstream.calls, [])
        self.assertEqual(self.call("/health", key="client-secret")[0], 200)
        self.assertTrue(all(row[2] is None for row in self.upstream.calls))

    def test_multiple_models_require_explicit_selection(self):
        self.upstream.models.append({"id": "other"})
        self.assertEqual(self.call("/health")[0], 503)
        adapter = httpd.Adapter(
            httpd.Config(endpoint=self.origin + "/v1/", model="resident")
        )
        self.assertEqual(adapter.model(time.monotonic() + 1)["id"], "resident")

    def test_bad_requests_are_client_errors(self):
        for body in (
            {"text": []},
            {"text": "Hallo", "target_language": "auto"},
            {"text": "Hallo", "max_tokens": True},
            [],
        ):
            with self.subTest(body=body):
                self.assertEqual(self.call("/translate", body)[0], 400)
        self.assertEqual(self.call("/translate", raw=b"{")[0], 400)
        self.assertEqual(self.call("/v1/chat/completions", {"stream": True})[0], 400)
        self.assertEqual(self.call("/arbitrate", {})[0], 501)
        self.assertEqual(self.call("/not-a-route", {})[0], 404)

    def test_small_context_is_reported_and_large_context_caps_output(self):
        self.upstream.models[0]["max_model_len"] = 2048
        self.assertEqual(
            self.call("/translate", {"text": "Hallo", "source_language": "de"})[0], 503
        )
        self.upstream.models[0]["max_model_len"] = 4096
        self.assertEqual(
            self.call(
                "/translate",
                {"text": "Hallo", "source_language": "de", "max_tokens": 99999},
            )[0],
            200,
        )
        self.assertEqual(self.upstream.calls[-1][1]["max_tokens"], 2048)


class PureTests(unittest.TestCase):
    def test_fragments_preserve_all_characters_and_are_utf8_bounded(self):
        for text in ("αβ🌎" * 99, "a b c\r\nd e f\n" * 100, "\t  日本語 \n"):
            for limit in (4, 9, 50, 1600):
                parts = list(httpd.fragments(text, limit))
                self.assertEqual("".join(parts), text)
                self.assertTrue(all(len(part.encode()) <= limit for part in parts))

    def test_invalid_configuration_fails_before_serving(self):
        for config in (
            httpd.Config(endpoint="ftp://host"),
            httpd.Config(endpoint="http://user:password@host"),
            httpd.Config(concurrency=0),
            httpd.Config(timeout=float("nan")),
        ):
            with self.assertRaises(ValueError):
                httpd.Adapter(config)


if __name__ == "__main__":
    unittest.main()
