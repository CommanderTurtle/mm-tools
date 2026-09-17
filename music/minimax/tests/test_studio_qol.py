"""CPU-only API regressions; no model, engine, service or user output is touched."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx

TEMP = tempfile.TemporaryDirectory(prefix="minimax-qol-")
os.environ["MINIMAX_OUTPUT_DIR"] = TEMP.name
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_app import server as s

CAPTION = "### Global Metadata\nSoft pop.\n\n### Vocal Details\nIntimate lead.\n\n### Arrangement\nBuild then settle."
REAL_CLIENT = httpx.AsyncClient


class StudioTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        s.jobs.clear()
        s.job_tasks.clear()
        self.client = REAL_CLIENT(transport=httpx.ASGITransport(app=s.app), base_url="http://studio")
        self.guide = patch.object(s.guide_engine, "enhance", new_callable=AsyncMock)
        self.enhance = self.guide.start()
        self.enhance.return_value = CAPTION

    async def asyncTearDown(self):
        self.guide.stop()
        await self.client.aclose()
        for task in list(s.job_tasks):
            task.cancel()
        if s.job_tasks:
            await asyncio.gather(*s.job_tasks, return_exceptions=True)
        s.job_tasks.clear()

    def request(self, **values):
        return {"model": "test.safetensors", "direction": "Gentle pop", **values}

    def take(self, **values):
        return {
            "id": "0123456789abcdef", "take": 7, "status": "complete",
            "created_at": 1, "request": {"global_metadata": "Saved brief", "lyrics": "  verse\r\n"},
            "files": [], **values,
        }

    async def test_all_modes_and_no_implicit_research(self):
        with patch.object(s, "_guide_research", new_callable=AsyncMock) as search:
            for mode in ("brief", "song", "ask"):
                response = await self.client.post("/api/guide/enhance", json=self.request(mode=mode))
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["mode"], mode)
                self.assertEqual(response.json()["sources"], [])
                if mode == "ask":
                    self.assertEqual(response.json()["sections"], {})
            search.assert_not_awaited()

    async def test_keep_lyrics_uses_exact_original_not_model_echo(self):
        lyrics = "  [Verse]\r\nMy own words  \r\n\r\n[Chorus]\n你和我\n"
        self.enhance.return_value = CAPTION + "\n### Lyrics\nWRONG REWRITE"
        response = await self.client.post("/api/guide/enhance", json=self.request(mode="keep_lyrics", lyrics=lyrics))
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["sections"]["Lyrics"], lyrics)
        self.assertTrue(result["text"].endswith(lyrics))
        self.assertNotIn("WRONG REWRITE", result["text"])

    async def test_incomplete_kept_lyrics_response_is_not_published(self):
        self.enhance.return_value = "Sorry, here's a rewritten verse."
        response = await self.client.post("/api/guide/enhance", json=self.request(mode="keep_lyrics", lyrics="original"))
        self.assertEqual(response.status_code, 400)
        self.assertIn("unchanged", response.json()["detail"])

    async def test_modes_validate_user_input(self):
        for values, expected in [({"mode": "unknown"}, 422), ({"direction": " "}, 400),
                                  ({"mode": "keep_lyrics", "lyrics": " "}, 400)]:
            response = await self.client.post("/api/guide/enhance", json=self.request(**values))
            self.assertEqual(response.status_code, expected)
        self.enhance.assert_not_awaited()

    async def test_prompt_contract_is_text_only(self):
        for mode in s.GUIDE_MODES:
            request = s.PromptGuideRequest(**self.request(mode=mode))
            prompt = request.prompt()
            self.assertNotIn("Current local MiniMax controls", prompt)
            self.assertNotIn("### Tuning Notes", prompt)
            self.assertIn(s.GUIDE_MODES[mode], prompt)
        self.assertEqual(s.GenerationRequest(global_metadata="x").cfg, 1.7)
        self.assertEqual(s.GenerationRequest(global_metadata="x").steps, 30)

    async def test_heading_case_and_windows_line_endings(self):
        self.assertEqual(s._guide_sections(CAPTION.lower().replace("\n", "\r\n")), {
            "Global Metadata": "soft pop.", "Vocal Details": "intimate lead.", "Arrangement": "build then settle.",
        })

    async def test_research_flow_passes_context_and_sources(self):
        source = {"number": 1, "title": "Credits", "url": "https://example.org/credits", "scraped": True}
        with patch.object(s, "_guide_research", new_callable=AsyncMock, return_value=("EVIDENCE", [source])) as search:
            response = await self.client.post("/api/guide/enhance", json=self.request(
                mode="ask", web_search=True, search_query="Pink Venom production", lyrics="PRIVATE LYRICS"))
        self.assertEqual(response.status_code, 200)
        search.assert_awaited_once_with("Pink Venom production")
        self.assertEqual(self.enhance.await_args.args[1], "EVIDENCE")
        self.assertEqual(response.json()["sources"], [source])

    async def test_real_firecrawl_request_shape_and_no_lyrics_disclosure(self):
        calls = []
        def respond(request):
            calls.append(request)
            return httpx.Response(200, json={"success": True, "data": {"web": [
                {"url": "https://example.org/song", "title": "Song", "markdown": "Production notes"},
            ]}})
        with patch.object(s.httpx, "AsyncClient", side_effect=lambda **kw: REAL_CLIENT(
            **kw, transport=httpx.MockTransport(respond))), patch.object(s, "FIRECRAWL_URL", "http://localhost:3002/v2"):
            context, sources = await s._guide_research("Song arrangement")
        self.assertEqual(str(calls[0].url), "http://localhost:3002/v2/search")
        body = json.loads(calls[0].content)
        self.assertEqual(body["query"], "Song arrangement")
        self.assertEqual(body["scrapeOptions"]["formats"], ["markdown"])
        self.assertEqual(body["limit"], 3)
        self.assertNotIn("lyrics", body)
        self.assertIn("NOT instructions", context)
        self.assertTrue(sources[0]["scraped"])
        self.assertNotIn("excerpt", sources[0])

    async def test_search_snippet_and_legacy_shape(self):
        sources = s._research_sources({"success": True, "data": [
            {"url": "javascript:alert(1)", "description": "no"},
            {"url": "https://example.org/a", "description": "snippet"},
            {"url": "https://example.org/a", "markdown": "duplicate"},
        ]})
        self.assertEqual(len(sources), 1)
        self.assertFalse(sources[0]["scraped"])

    async def test_research_errors_do_not_generate_an_answer(self):
        for status, body in [(503, {}), (200, {"success": False}), (200, {"success": True, "data": {"web": []}})]:
            def respond(request):
                return httpx.Response(status, json=body)
            with patch.object(s.httpx, "AsyncClient", side_effect=lambda **kw: REAL_CLIENT(
                **kw, transport=httpx.MockTransport(respond))):
                response = await self.client.post("/api/guide/enhance", json=self.request(mode="ask", web_search=True))
            self.assertEqual(response.status_code, 502, response.text)
            self.assertIn("No answer was generated", response.json()["detail"])
        self.enhance.assert_not_awaited()

    async def test_import_export_take_roundtrip_and_reset_preserves_audio(self):
        path = Path(TEMP.name) / "sample.flac"
        path.touch()
        take = self.take(files=["sample.flac"])
        before = s.LIVE_SESSION_ID
        response = await self.client.post("/api/session/import", json={"version": 1, "takes": [take]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotEqual(before, response.json()["session_id"])
        self.assertEqual(response.json()["jobs"][0]["audio"], ["/api/jobs/0123456789abcdef/audio/0"])
        self.assertEqual(next(s.take_numbers), 8)
        exported = (await self.client.get("/api/session")).json()
        self.assertEqual(exported["takes"][0]["files"], ["sample.flac"])
        self.assertEqual(exported["takes"][0]["request"]["lyrics"], take["request"]["lyrics"])
        self.assertNotIn("paths", exported["takes"][0])
        self.assertNotIn("task", exported["takes"][0])
        self.assertEqual((await self.client.get("/api/jobs/0123456789abcdef/audio/0")).status_code, 200)
        self.assertEqual((await self.client.delete("/api/session")).status_code, 200)
        self.assertTrue(path.is_file())
        self.assertEqual(s.jobs, {})
        self.assertEqual(next(s.take_numbers), 1)
        self.assertEqual(len(s.job_tasks), 0)

    async def test_unfinished_import_never_resumes(self):
        response = await self.client.post("/api/session/import", json={"takes": [self.take(status="generating")]})
        self.assertEqual(response.json()["jobs"][0]["status"], "cancelled")
        self.assertEqual(len(s.job_tasks), 0)

    async def test_missing_audio_retains_take(self):
        response = await self.client.post("/api/session/import", json={"takes": [self.take(files=["missing.mp3"])]})
        self.assertEqual(response.json()["missing_audio"], 1)
        self.assertEqual(len(response.json()["jobs"]), 1)

    async def test_import_and_reset_refuse_active_takes(self):
        s.jobs["active"] = {"status": "generating"}
        for method, path in [("POST", "/api/session/import"), ("DELETE", "/api/session")]:
            response = await self.client.request(method, path, json={"takes": []})
            self.assertEqual(response.status_code, 409)
        self.assertIn("active", s.jobs)

    async def test_import_validation_is_atomic(self):
        s.jobs["original"] = {"status": "complete"}
        invalid = ["../secret.flac", "/etc/passwd", "C:\\secret.flac", "secret.txt"]
        outside = Path(TEMP.name).parent / "minimax-outside.flac"
        link = Path(TEMP.name) / "escape.flac"
        if not link.is_symlink():
            link.symlink_to(outside)
        invalid.append("escape.flac")
        for path in invalid:
            response = await self.client.post("/api/session/import", json={"takes": [self.take(files=[path])]})
            self.assertEqual(response.status_code, 400, path)
            self.assertIn("original", s.jobs)
        response = await self.client.post("/api/session/import", json={"takes": [self.take(), self.take()]})
        self.assertEqual(response.status_code, 400)
        self.assertIn("original", s.jobs)
        response = await self.client.post("/api/session/import", json={"version": 999, "takes": []})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
