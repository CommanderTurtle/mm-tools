from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from studio.runtime import JobRunner, JobStore, StudioAdapter, StudioContext, StudioOutput


class FakeAdapter(StudioAdapter):
    def validate(self, request, resolve_asset):
        if request.get("mode") != "receipt":
            raise ValueError("unsupported test mode")
        asset_id = request.get("controls", {}).get("asset")
        if asset_id:
            resolve_asset(asset_id)

    def run(self, request, context: StudioContext):
        context.update("Writing receipt", 0.5, "fake adapter entered")
        path = context.output_dir / "receipt.json"
        path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
        context.update("Receipt ready", 1.0)
        return [StudioOutput(path, "document", "Request receipt", "application/json")]

    def load(self):
        return {"ready": True, "loaded": True, "details": []}


class StudioRuntimeTests(unittest.TestCase):
    def test_asset_job_output_and_persistence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            store = JobStore(runtime / "studio.sqlite3")
            adapter = FakeAdapter(root, runtime)
            runner = JobRunner(store, adapter, runtime)
            try:
                source = runtime / "assets" / "asset.txt"
                source.write_text("local-only", encoding="utf-8")
                store.add_asset("asset-1", "example.txt", source.name, "text/plain", source.stat().st_size)

                request = {"mode": "receipt", "controls": {"asset": "asset-1", "seed": 7}}
                job = runner.submit(request)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    job = store.get_job(job["id"])
                    if job["status"] in {"complete", "failed", "cancelled", "interrupted"}:
                        break
                    time.sleep(0.01)

                self.assertEqual(job["status"], "complete", job.get("error"))
                self.assertEqual(job["progress"], 1.0)
                self.assertEqual(len(job["outputs"]), 1)
                output = job["outputs"][0]
                self.assertEqual(output["relative"], "receipt.json")
                self.assertEqual(output["media_type"], "application/json")
                persisted = json.loads((runtime / "outputs" / job["id"] / output["relative"]).read_text())
                self.assertEqual(persisted, request)
                self.assertTrue(any(item["message"] == "fake adapter entered" for item in job["log"]))
                self.assertTrue(runner.model_action("load")["loaded"])

                reopened = JobStore(runtime / "studio.sqlite3")
                self.assertEqual(reopened.get_job(job["id"])["status"], "complete")
                self.assertEqual(reopened.get_asset("asset-1")["name"], "example.txt")
            finally:
                runner.close()

    def test_rejects_invalid_mode_before_creating_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            store = JobStore(runtime / "studio.sqlite3")
            runner = JobRunner(store, FakeAdapter(root, runtime), runtime)
            try:
                with self.assertRaisesRegex(ValueError, "unsupported test mode"):
                    runner.submit({"mode": "wrong", "controls": {}})
                self.assertEqual(store.list_jobs(), [])
            finally:
                runner.close()


if __name__ == "__main__":
    unittest.main()
