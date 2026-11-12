"""Validate local artifacts and package the studio-owned viewer, not agent HTML."""

import json
import math
from pathlib import Path
import shutil
import struct
import zipfile
import xml.etree.ElementTree as ET

from state import ROOT, atomic, inside, read

PATHS = {
    "input": {".png"},
    "model": {".glb"},
    "blend": {".blend"},
    "simulator": {".xml"},
    "video": {".mp4", ".webm"},
    "native_render": {".png", ".jpg"},
    "refined_render": {".png", ".jpg"},
    "physics_report": {".json"},
    "native_preservation_evidence": {".json"},
}


def glb_info(path):
    with Path(path).open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12 or struct.unpack("<III", header)[:2] != (0x46546C67, 2):
            raise ValueError("Export must be GLB 2.0")
        if struct.unpack("<III", header)[2] != Path(path).stat().st_size:
            raise ValueError("GLB length does not match its file")
        size, kind = struct.unpack("<II", stream.read(8))
        if kind != 0x4E4F534A or size > 64 * 1024 * 1024:
            raise ValueError("Missing or oversized GLB JSON chunk")
        data = json.loads(stream.read(size))
    unsupported = {
        "KHR_draco_mesh_compression",
        "EXT_meshopt_compression",
        "KHR_meshopt_compression",
        "KHR_texture_basisu",
    }
    if unsupported.intersection(data.get("extensionsRequired", [])):
        raise ValueError(
            "Export GLB without Draco, Meshopt or KTX2 compression for the local viewer"
        )
    for row in data.get("buffers", []) + data.get("images", []):
        if row.get("uri") and not row["uri"].startswith("data:"):
            raise ValueError("GLB must embed its buffers and textures")
    return data


def interaction(work, model):
    value = read(work / "interaction.json", {"joints": []})
    if not isinstance(value, dict) or not isinstance(value.get("joints"), list):
        raise ValueError("interaction.json needs a joints array")
    if len(value["joints"]) > 256:
        raise ValueError("Too many direct joint controls")
    names = [n.get("name") for n in model.get("nodes", [])]
    used = set()
    for joint in value["joints"]:
        if not isinstance(joint, dict) or joint.get("kind") not in (
            "rotate",
            "translate",
        ):
            raise ValueError("Joint kind must be rotate or translate")
        node = joint.get("node")
        if not isinstance(node, str) or names.count(node) != 1 or node in used:
            raise ValueError("Joint must reference a unique GLB node: " + str(node))
        used.add(node)
        if not isinstance(joint.get("label"), str) or len(joint["label"]) > 160:
            raise ValueError("Joint label is missing or too long")
        axis = joint.get("axis")
        numbers = [joint.get(k) for k in ("min", "max", "initial")]
        if not isinstance(axis, list) or len(axis) != 3:
            raise ValueError("Joint axis needs three numbers")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) for v in axis + numbers
        ):
            raise ValueError("Joint values must be finite")
        if (
            sum(v * v for v in axis) < 1e-12
            or not numbers[0] <= numbers[2] <= numbers[1]
            or numbers[0] == numbers[1]
        ):
            raise ValueError("Joint range/axis is invalid")
        joint["unit"] = "deg" if joint["kind"] == "rotate" else "m"
    return value


def check_mjcf(work, path, visited=None, inherited=None):
    visited = set() if visited is None else visited
    if path in visited:
        return
    visited.add(path)
    xml = ET.parse(path).getroot()
    compiler = xml.find("compiler")
    dirs = dict(inherited or {})
    if compiler is not None:
        dirs.update(compiler.attrib)
    for node in xml.iter():
        name = node.get("file")
        if not name:
            continue
        if Path(name).is_absolute() or "://" in name:
            raise ValueError("MJCF contains an external file: " + name)
        prefix = dirs.get("assetdir", "")
        if node.tag == "mesh":
            prefix = dirs.get("meshdir", prefix)
        elif node.tag == "texture":
            prefix = dirs.get("texturedir", prefix)
        elif node.tag == "include":
            prefix = ""
        dependency = inside(work, path.parent / prefix / name)
        if not dependency.is_file() or not dependency.stat().st_size:
            raise ValueError("Missing MJCF dependency: " + name)
        if node.tag == "include":
            check_mjcf(work, dependency, visited, dirs)


def validate(work):
    data = read(work / "delivery.json")
    if not isinstance(data, dict):
        raise ValueError(
            "The agent did not write delivery.json; partial files are retained"
        )
    if not isinstance(data.get("title"), str) or not 0 < len(data["title"]) <= 160:
        raise ValueError("Delivery needs a title up to 160 characters")
    enums = {
        "status": {
            "delivered",
            "delivered_with_limitations",
            "native_only",
            "blocked",
            "failed",
        },
        "motion_kind": {
            "simulated_replay",
            "kinematic_parameter_animation",
            "static_only",
        },
        "physics_status": {"passed", "failed", "partial", "not_tested"},
        "interaction_status": {"bound_motion_verified", "partial", "static_only"},
    }
    for key, values in enums.items():
        if data.get(key) not in values:
            raise ValueError("Invalid delivery " + key)
    for key in ("acceptance_passes", "repair_rounds"):
        if type(data.get(key)) is not int or not 0 <= data[key] <= 2:
            raise ValueError("Delivery exceeded the bounded " + key)
    if data["status"] == "blocked" and not data.get("blocker"):
        raise ValueError("Blocked delivery needs a reason")
    for key in ("limitations", "reference_code_used", "visible_motion_evidence"):
        if not isinstance(data.get(key), list) or any(
            not isinstance(v, str) for v in data[key]
        ):
            raise ValueError("Delivery needs a " + key + " array")
    for key, suffixes in PATHS.items():
        name = data.get(key)
        if name is None:
            continue
        if not isinstance(name, str) or Path(name).is_absolute():
            raise ValueError("Delivery paths must be relative")
        file = inside(work, name)
        if (
            file.suffix.lower() not in suffixes
            or not file.is_file()
            or not file.stat().st_size
        ):
            raise ValueError("Missing or invalid delivery file: " + key)
    for name in data["visible_motion_evidence"]:
        file = inside(work, name)
        if (
            not file.is_file()
            or file.suffix.lower() not in (".png", ".jpg")
            or not file.stat().st_size
        ):
            raise ValueError("Missing visible motion evidence: " + name)
    model = glb_info(inside(work, data["model"])) if data.get("model") else {}
    controls = interaction(work, model) if model else {"joints": []}
    moving = any(a.get("channels") for a in model.get("animations", []))
    if data["motion_kind"] != "static_only" and not moving:
        raise ValueError("Claimed motion needs real GLB animation channels")
    if data["interaction_status"] == "bound_motion_verified" and (
        not moving or len(data["visible_motion_evidence"]) < 2
    ):
        raise ValueError("Verified interaction needs motion and rendered evidence")
    if data.get("simulator"):
        check_mjcf(work, inside(work, data["simulator"]))
    if data["physics_status"] == "passed" and not (
        data.get("simulator")
        and data.get("physics_report")
        and data["motion_kind"] == "simulated_replay"
    ):
        raise ValueError(
            "Physics passed needs a simulator, report and simulated replay"
        )
    if data["status"] == "delivered" and not all(data.get(k) for k in PATHS):
        raise ValueError("A complete delivery needs the complete artifact set")
    atomic(
        work / "validated.json",
        {
            "delivery": data,
            "interaction": controls,
            "clips": [a.get("name", "Action") for a in model.get("animations", [])],
            "verification": "Artifact structure checked; physics measurements are agent-reported.",
        },
    )
    return data


def package(work):
    if work.is_symlink() or any(p.is_symlink() for p in work.rglob("*")):
        raise ValueError("Export files must be local, not symlinks")
    data = validate(work)
    page = work / "presentation"
    page.mkdir(exist_ok=True)
    shutil.copy2(ROOT.parents[1] / "LICENSE", page / "LICENSE")
    for name in ("viewer.html", "view.js", "player.js", "viewer.css"):
        shutil.copy2(
            ROOT / "web" / name,
            page / ("index.html" if name == "viewer.html" else name),
        )
    three = ROOT / "node_modules/three"
    for name in (
        "build/three.module.js",
        "build/three.core.js",
        "examples/jsm/loaders/GLTFLoader.js",
        "examples/jsm/controls/OrbitControls.js",
        "examples/jsm/environments/RoomEnvironment.js",
        "examples/jsm/utils/BufferGeometryUtils.js",
        "examples/jsm/utils/SkeletonUtils.js",
        "LICENSE",
    ):
        target = page / "vendor" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(three / name, target)
    atomic(
        page / "scene.json",
        {"delivery": data, "interaction": read(work / "validated.json")["interaction"]},
    )
    target = work.parent / "asset.zip"
    pending = target.with_suffix(".partial")
    try:
        with zipfile.ZipFile(
            pending, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3
        ) as archive:
            for file in work.rglob("*"):
                if file.is_symlink():
                    raise ValueError(
                        "Export contains a symlink: "
                        + file.relative_to(work).as_posix()
                    )
                if file.is_file() and ".partial" not in file.name:
                    archive.write(file, file.relative_to(work).as_posix())
        pending.replace(target)
    finally:
        pending.unlink(missing_ok=True)
    return data
