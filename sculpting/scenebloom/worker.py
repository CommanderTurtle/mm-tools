import logging
import os
import subprocess
import sys
import traceback
import uuid
from lifecycle import own, release
from state import NATIVE, WORK, LANE_STAMP, atomic, read, job_path
from native import install


def execute(job, stage):
    # Native variant selection scans the proposal tree. Keep failed attempts out
    # of that tree so resuming cannot mix two independently sampled layouts.
    folder = {"proposals": "proposals", "render": "renders"}.get(stage)
    if folder and (job / folder).exists():
        archive = job / "attempts"
        archive.mkdir(exist_ok=True)
        (job / folder).rename(archive / f"{stage}-{uuid.uuid4().hex}")
    install()
    from scenebloom.utils.config import PipelineConfig
    from scenebloom.utils.schemas import SceneSpec, SceneRecord, ZoningPlan, SceneLayout
    from scenebloom.pipeline import PipelineOrchestrator

    data = read(job / "run.json")
    options = {k: v for k, v in data["settings"].items() if k != "randomize"}
    config = PipelineConfig(
        **options,
        output_dir=job,
        embed_index_dir=WORK / "index/front",
        embed_index_extra_dirs=(WORK / "index/mp3d",),
    )
    pipeline = PipelineOrchestrator(config)

    def load(name, cls):
        return cls.model_validate(read(job / name))

    def save(name, model):
        atomic(job / name, model.model_dump(mode="json"))

    if stage == "index":
        base = NATIVE / "data/m3dlayout/extracted/layout/M3DLayout_json"
        for name, prefix, flags in (
            ("front", "3dfront", []),
            (
                "mp3d",
                "mp3d",
                [
                    "--room-types",
                    "office",
                    "bathroom",
                    "kitchen",
                    "lounge",
                    "rec_game",
                    "meeting",
                ],
            ),
        ):
            folder = WORK / "index" / name
            if read(folder / "complete.json", {}).get("source") == LANE_STAMP and all(
                (folder / p).is_file() for p in ("embeddings.npy", "metadata.json")
            ):
                continue
            subprocess.run(
                [
                    sys.executable,
                    str(NATIVE / "scripts/build_index/build_embeddings.py"),
                    str(folder),
                    *[
                        str(base / f"{prefix}_{split}.json")
                        for split in ("train", "val", "test")
                    ],
                    *flags,
                ],
                env=dict(os.environ, PYTHONPATH=str(NATIVE)),
                check=True,
            )
            atomic(folder / "complete.json", dict(source=LANE_STAMP))
    elif stage == "retrieve":
        from scenebloom.sampling.zoning.prompt_parser import PromptParser
        from scenebloom.render.materials import supported_room_type

        spec = PromptParser().parse(data["prompt"])
        if not spec.room_constraints or not supported_room_type(
            spec.room_constraints[0].room_type
        ):
            raise ValueError(
                "Choose a bedroom, living room, dining room, office or bathroom"
            )
        save("spec.json", spec)
        atomic(
            job / "references.json",
            [s.model_dump(mode="json") for s in pipeline._retrieve(spec)],
        )
    elif stage == "zone":
        from scenebloom.sampling.zoning.llm_zone_decomposer import LLMZoneDecomposer

        refs = [SceneRecord.model_validate(r) for r in read(job / "references.json")]
        plan = LLMZoneDecomposer(
            config.m3dlayout_root, top_k_scenes=config.retrieval_context_n
        ).decompose(load("spec.json", SceneSpec), refs)
        save("plan.json", plan)
    elif stage == "proposals":
        from scenebloom.utils.assets import AssetDB, AssetSelector

        plan = load("plan.json", ZoningPlan)
        candidates = AssetSelector(
            AssetDB.from_path(config.assets_path)
        ).select_for_plan(plan)
        save("lifted.json", pipeline._strategy.generate(plan, candidates))
    elif stage == "compile":
        from scenebloom.compilation.compiler import compile_scene_from_pipeline

        save(
            "compiled.json",
            compile_scene_from_pipeline(config, load("lifted.json", SceneLayout)),
        )
    elif stage == "complete":
        from scenebloom.pipeline import (
            _assign_asset_variants,
            _place_wall_objects,
            _place_small_objects,
        )
        from scenebloom.utils.assets import AssetDB

        assets = AssetDB.from_path(config.assets_path)
        layout = _assign_asset_variants(config, load("compiled.json", SceneLayout))
        layout = _place_wall_objects(config, layout, assets)
        save("scene_layout.json", _place_small_objects(config, layout, assets))
    elif stage == "materials":
        pipeline._texture_assignment(load("scene_layout.json", SceneLayout))
    elif stage == "render":
        from scenebloom.render.renderer import GenesisRenderer

        renderer = GenesisRenderer(material_assignment=job / "material_assignment.json")
        renderer.load_scene(load("scene_layout.json", SceneLayout))
        renderer.render(job / "renders")
    elif stage == "export":
        from scenebloom.export import export_scene

        export_scene(
            load("scene_layout.json", SceneLayout),
            job / "material_assignment.json",
            job / "scene.glb",
        )
    else:
        raise ValueError("Unknown stage")
    atomic(job / f"{stage}.done.json", dict(source=LANE_STAMP))


if __name__ == "__main__":
    key, stage, parent = sys.argv[1:]
    own(int(parent))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    job = job_path(key)
    try:
        execute(job, stage)
    except Exception as error:
        atomic(job / "error.json", dict(error=str(error), stage=stage))
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        release()
