import os

from state import ROOT, MODELS, NATIVE, require_models, source_paths, read

DISPATCH = "groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu"


def grounding_dispatch(path):
    """Torch 2.7 dispatches on ScalarType; the pinned source still says value.type()."""
    target = path / DISPATCH
    if not target.is_file():
        raise SystemExit("Missing GroundingDINO dispatch source: " + str(target))
    original = target.read_bytes()
    before = b"AT_DISPATCH_FLOATING_TYPES(value.type(),"
    after = b"AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),"
    if original.count(before) == 0 and original.count(after) == 2:
        return None  # an earlier setup run already applied the torch 2.7 patch
    if original.count(before) != 2:
        raise SystemExit("GroundingDINO dispatch changed; review the source pin")
    return target, original.replace(before, after)


def link(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() and path.resolve() == target.resolve():
        return
    if path.exists() or path.is_symlink():
        raise SystemExit(f"Preserving existing path: {path}; expected link to {target}")
    path.symlink_to(
        os.path.relpath(target, path.parent), target_is_directory=target.is_dir()
    )


def local_links():
    """Link the committed trees into the paths SceneBloom resolves."""
    link(NATIVE / "local_app/ovmono3d", ROOT / "ovmono3d")
    trellis = ROOT.parent / "trellis"
    if trellis.is_dir():
        # text-to-mesh runs from the shared TRELLIS lane beside this one
        link(NATIVE / "local_app/TRELLIS", trellis)


def main():
    local_links()
    require_models()
    entries = {
        "scenebloom": "scenebloom/__init__.py",
        "ovmono3d": "cubercnn/config/__init__.py",
        "dinov2": "dinov2/__init__.py",
        "pytorch3d": "pytorch3d/__init__.py",
        "detectron2": "detectron2/__init__.py",
        "groundingdino": "groundingdino/__init__.py",
        "depthpro": "src/depth_pro/__init__.py",
        "sam": "segment_anything/__init__.py",
    }
    for path, revision in source_paths():
        name = path.name
        if not (path / entries[name]).is_file():
            raise SystemExit(f"Missing committed native source: {path / entries[name]}")
        if not revision.startswith(tuple("0123456789abcdef")):
            raise SystemExit(f"Malformed native revision for {path}")
        if name == "groundingdino":
            compatibility = grounding_dispatch(path)
            if compatibility:
                compatibility[0].write_bytes(compatibility[1])
    for path, target in (
        (ROOT / "models", MODELS),
        (NATIVE / "checkpoints", MODELS / "checkpoints"),
        (ROOT / "ovmono3d/checkpoints", MODELS / "checkpoints/ovmono3d"),
        (NATIVE / "data/lora_assets/glb", MODELS / "assets/glb"),
        (NATIVE / "data/lora_assets/adapters", MODELS / "assets/adapters"),
        (NATIVE / "data/m3dlayout", MODELS / "m3dlayout"),
        (NATIVE / "data/materials/ambientcg_full", MODELS / "materials"),
    ):
        link(path, target)
    library = read(NATIVE / "data/lora_assets/instances.json")
    expected = [
        MODELS
        / "assets/adapters"
        / row["output_name"]
        / "pytorch_lora_weights.safetensors"
        for row in library["categories"] + library["variants"]
        if "output_name" in row
    ]
    expected += [
        MODELS / "assets/glb" / f"{key}.glb"
        for key in read(NATIVE / "data/meshes.json")
    ]
    missing = [str(p) for p in expected if not p.is_file()]
    if missing:
        raise SystemExit(
            "Complete the scenebloom model bundle: " + ", ".join(missing[:5])
        )


if __name__ == "__main__":
    main()
