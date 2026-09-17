from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECTS = (
    "4d-ify",
    "V2V",
    "animate",
    "aukspeech",
    "music/yue2",
    "nvidia-sim",
    "sculpting",
    "sculpting/world",
    "world-gen/fire3d",
    "world-gen/lingbot",
)
SUPPORTED_FIELDS = {
    "asset",
    "choice",
    "code",
    "collection",
    "date",
    "emotion_mixer",
    "multi_asset",
    "multi_choice",
    "nonverbal_events",
    "number",
    "range",
    "section",
    "select",
    "tags",
    "text",
    "textarea",
    "toggle",
}
OFFLINE_MARKERS = (
    "HF_HUB_DISABLE_TELEMETRY=1",
    "HF_HUB_OFFLINE=1",
    "TRANSFORMERS_OFFLINE=1",
    "DO_NOT_TRACK=1",
)


def manifest_for(project: str) -> dict:
    path = ROOT / project / "local_app" / "studio.json"
    return json.loads(path.read_text(encoding="utf-8"))


def resolved_fields(manifest: dict, mode: dict) -> list[dict]:
    fields: list[dict] = []
    for zone in ("fields", "advanced"):
        for reference in mode.get(f"{zone}_refs", []):
            fields.extend(manifest["field_sets"][reference])
        fields.extend(mode.get(zone, []))
    return fields


class StudioContracts(unittest.TestCase):
    def test_manifests_and_adapter_routes(self) -> None:
        seen_ids: set[str] = set()
        # MiniMax owns one public port plus two loopback-only engines.
        seen_ports: set[int] = {8254, 8255, 8256}
        for project in PROJECTS:
            with self.subTest(project=project):
                root = ROOT / project
                manifest = manifest_for(project)
                self.assertTrue(manifest["id"])
                self.assertNotIn(manifest["id"], seen_ids)
                seen_ids.add(manifest["id"])

                port = int(manifest["port"])
                self.assertGreaterEqual(port, 1024)
                self.assertNotIn(port, seen_ports)
                seen_ports.add(port)

                mode_ids = [mode["id"] for mode in manifest["modes"]]
                self.assertEqual(len(mode_ids), len(set(mode_ids)))
                self.assertGreater(len(mode_ids), 0)

                adapter_path = root / "local_app" / "adapter.py"
                adapter_source = adapter_path.read_text(encoding="utf-8")
                ast.parse(adapter_source, filename=str(adapter_path))
                constants = {
                    node.value
                    for node in ast.walk(ast.parse(adapter_source))
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)
                }
                self.assertEqual(set(mode_ids) - constants, set())

                for mode in manifest["modes"]:
                    references = mode.get("fields_refs", []) + mode.get("advanced_refs", [])
                    self.assertEqual(
                        set(references) - set(manifest.get("field_sets", {})),
                        set(),
                        f"{mode['id']} refers to an unknown field set",
                    )
                    fields = resolved_fields(manifest, mode)
                    ids = [field["id"] for field in fields if field.get("id")]
                    self.assertEqual(len(ids), len(set(ids)), f"duplicate fields in {mode['id']}")
                    available = set(ids)
                    self.assertEqual(set(mode.get("requires_any", [])) - available, set())
                    for field in fields:
                        self.assertIn(field["type"], SUPPORTED_FIELDS)
                        condition = field.get("show_if")
                        if condition:
                            self.assertIn(condition["field"], available)

    def test_setup_start_and_offline_contracts(self) -> None:
        for project in PROJECTS:
            with self.subTest(project=project):
                root = ROOT / project
                for relative in (
                    "ARCHITECTURE.md",
                    "setupwithuv.sh",
                    "startwithuv.sh",
                    "local_app/adapter.py",
                    "local_app/studio.json",
                ):
                    self.assertTrue((root / relative).is_file(), f"missing {project}/{relative}")
                start = (root / "startwithuv.sh").read_text(encoding="utf-8")
                self.assertIn("studio/server.py", start.replace("-m studio.server", "studio/server.py"))
                for marker in OFFLINE_MARKERS:
                    self.assertIn(marker, start)
                self.assertIn(str(manifest_for(project)["port"]), start)

    def test_browser_implements_every_manifest_field(self) -> None:
        javascript = (ROOT / "studio" / "web" / "app.js").read_text(encoding="utf-8")
        for field_type in SUPPORTED_FIELDS:
            self.assertIn(f'"{field_type}"', javascript)

        html = (ROOT / "studio" / "web" / "index.html").read_text(encoding="utf-8")
        ids = set(re.findall(r'\bid="([^"]+)"', html))
        referenced = set(re.findall(r'\$\("([^"]+)"\)', javascript))
        self.assertEqual(referenced - ids, set())

        css = (ROOT / "studio" / "web" / "style.css").read_text(encoding="utf-8")
        for source in (html, css):
            self.assertNotRegex(source, r"https?://")

    def test_relocated_minimax_contract(self) -> None:
        minimax = ROOT / "music" / "minimax"
        environment = (minimax / ".env.local.example").read_text(encoding="utf-8")
        self.assertIn("MINIMAX_MODEL_DIR=../../models/", environment)
        self.assertIn("MINIMAX_GUIDE_MODEL_ROOT=../../models/qwen", environment)
        self.assertIn("MINIMAX_ENGINE_PORT=8255", environment)
        self.assertIn("MINIMAX_GUIDE_ENGINE_PORT=8256", environment)

        launcher = (minimax / "startwithuv.sh").read_text(encoding="utf-8")
        self.assertIn('exec "$ROOT/.venv/bin/python" -m uvicorn', launcher)
        self.assertNotIn("source .venv/bin/activate", launcher)

        server = (minimax / "local_app" / "server.py").read_text(encoding="utf-8")
        self.assertIn('ROOT.parents[1] / "models"', server)

    def test_all_owned_python_parses(self) -> None:
        roots = [ROOT / "studio"] + [ROOT / project / "local_app" for project in PROJECTS]
        for folder in roots:
            for path in folder.rglob("*.py"):
                with self.subTest(path=path.relative_to(ROOT).as_posix()):
                    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


if __name__ == "__main__":
    unittest.main()
