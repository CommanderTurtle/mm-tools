"""Capture PHC's own rollout and collision geometry in its isolated interpreter."""

import importlib.util
import shutil
import sys
import types
import xml.etree.ElementTree as ET

import numpy as np

from state import SOURCE, atomic, read


def geometry(xml_path, names):
    result = []
    for body in ET.parse(xml_path).iter("body"):
        name = body.get("name")
        if name not in names:
            continue
        for geom in body.findall("geom"):
            kind = geom.get("type", "sphere")
            if kind not in ("sphere", "capsule", "box", "cylinder", "ellipsoid"):
                continue
            values = {
                key: [float(x) for x in geom.get(key).split()]
                for key in ("size", "pos", "quat", "fromto")
                if geom.get(key)
            }
            result.append(dict(body=names.index(name), type=kind, **values))
    return result


def reconstruct_physics(path, config, progress, cancel):
    # The native module imports Isaac Gym before Torch, as required by Gym.
    spec = importlib.util.spec_from_file_location(
        "flowhmr_phc_native", str(SOURCE / "tools/phc_score_server.py")
    )
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    scene = read(path / "viewer/scene.json")
    frames = scene["frames"]
    if frames < 2:
        raise ValueError("Physics tracking requires at least two frames")
    ctrl = path / "physics-control"
    ctrl.mkdir(exist_ok=True)
    xml_copy = ctrl / "humanoid.xml"

    from phc.env.tasks.humanoid import Humanoid

    original_create = Humanoid._create_smpl_humanoid_xml

    def capture_asset(task, *args, **kwargs):
        result = original_create(task, *args, **kwargs)
        if isinstance(result, dict) and 0 in result:
            shutil.copy2(result[0][1], xml_copy)
        return result

    def once(player):
        task = player.env.task
        task._sample_ref_state = types.MethodType(
            native._deterministic_sample_ref_state, task
        )
        task._termination_distances[:] = 0.5
        task._recovery_episode_prob = 0
        task._fall_init_prob = 0
        task.zero_out_far = False
        task.zero_out_far_train = False
        task.cycle_motion = False
        converter = native._make_converter(frames)
        motion = converter.convert_file(str(path / scene["motion"]))
        native._install_motions(task, {"motion": motion})
        native._load_batch(task, 0)
        positions, rotations, reference = [], [], []
        original_step = player.env_step

        def step(*args, **kwargs):
            cancel()
            response = original_step(*args, **kwargs)
            info = response[3]
            positions.append(native._as_numpy(info["body_pos"][0]).copy())
            rotations.append(native._as_numpy(task._rigid_body_rot[0]).copy())
            reference.append(native._as_numpy(info["body_pos_gt"][0]).copy())
            progress(
                "physics",
                min(99, len(positions) * 100 // frames),
                "Tracking native humanoid",
            )
            return response

        player.env_step = step
        try:
            metrics = native._rollout_one_batch(task, player, 0.15, 0.5)["motion"]
        finally:
            player.env_step = original_step
        skeleton = task.skeleton_trees[0]
        names = list(skeleton.node_names)
        parents = native._as_numpy(skeleton.parent_indices).astype(int).tolist()
        positions = np.asarray(positions, dtype=np.float32)
        rotations = np.asarray(rotations, dtype=np.float32)
        reference = np.asarray(reference, dtype=np.float32)
        if (
            not len(positions)
            or not np.isfinite(positions).all()
            or not np.isfinite(rotations).all()
        ):
            raise ValueError("PHC produced no finite rollout")
        if positions.shape[1] != len(names) or not xml_copy.is_file():
            raise ValueError("PHC body geometry does not match the recorded rollout")
        fps = 1.0 / float(task.dt)
        np.savez(
            path / "viewer/physics.npz",
            positions=positions,
            rotations=rotations,
            reference=reference,
            parents=np.asarray(parents),
            names=np.asarray(names),
            fps=fps,
        )
        atomic(
            path / "viewer/physics.json",
            dict(
                positions=positions.tolist(),
                rotations=rotations.tolist(),
                reference=reference.tolist(),
                parents=parents,
                names=names,
                fps=fps,
                up_axis="z",
                first_time=float(task.dt),
                geometry=geometry(xml_copy, names),
                metrics=metrics,
            ),
        )
        progress("ready", 100, "Physics tracking saved")

    native._server_run = once
    sys.argv = [
        str(SOURCE / "tools/phc_score_server.py"),
        "--ctrl-dir",
        str(ctrl),
        "--num-envs",
        "1",
        "--device-id",
        str(config["device"]),
        "--max-frames",
        str(frames),
    ]
    # Capture the asset actually loaded by Isaac Gym, not the converter's rig.
    Humanoid._create_smpl_humanoid_xml = capture_asset
    try:
        native.main()
    finally:
        Humanoid._create_smpl_humanoid_xml = original_create
