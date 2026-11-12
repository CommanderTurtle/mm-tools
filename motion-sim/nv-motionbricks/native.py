"""MotionBricks adapter: inference, locomotion, IK and overlap rules stay upstream."""

import json
import math
import uuid

import newton
import numpy as np
import warp as wp

from app.runtime.common.rig import load_rig_assets, model_to_runtime
from app.runtime.common.io.smart_primitive_document import (
    SmartPrimitiveAssetPool,
    load_smart_primitive_document,
)
from app.runtime.common.io.scene_document import (
    load_scene_document,
    check_reference_rigs,
)
from app.runtime.common.motion import LocomotionCommand
from app.runtime.common.smart_primitive_behaviors import behavior_types
from app.runtime.soma.character import SOMACharacter
from app.runtime.unitree_g1.character import UnitreeG1Character
from motionbricks.inference.builder import TensorRTBuildOptions
from motionbricks.model.contract import ModelRig
from motionbricks.model.config import MotionBricksInferenceConfig
from motionbricks.model.soma.engine import MotionBricksSOMAEngine
from motionbricks.model.soma.contract import (
    SOMA_N1_FP32_ONNX_PATH,
    SOMA_N1_FP16_ONNX_PATH,
)
from motionbricks.model.unitree_g1.engine import MotionBricksUnitreeG1Engine
from motionbricks.model.unitree_g1.contract import (
    UNITREE_G1_N1_FP32_ONNX_PATH,
    UNITREE_G1_N1_FP16_ONNX_PATH,
)
from motionbricks.runtime.smart_primitive import (
    InteractionBox,
    compute_interaction_box_overlaps,
)
from motionbricks.renderers.foot_contact_renderer import (
    skeletal_mesh_sole_outlines,
    body_sole_outlines,
)
from motionbricks.model.soma.contract import SOMA_N1_FOOT_CONTACT_JOINTS
from motionbricks.model.unitree_g1.contract import (
    UNITREE_G1_N1_FOOT_CONTACT_BODIES,
    UNITREE_G1_N1_FOOT_CONTACT_OFFSETS,
)

from storage import ROOT, WORK, resolve, virtual, read_document, refs, write_document


def transform(value):
    position = np.asarray(value.get("translation", [0, 0, 0]), dtype=float)
    rotation = np.asarray(value.get("rotation_xyzw", [0, 0, 0, 1]), dtype=float)
    if (
        position.shape != (3,)
        or rotation.shape != (4,)
        or not np.isfinite(np.r_[position, rotation]).all()
        or np.linalg.norm(rotation) < 1e-8
    ):
        raise ValueError("Expected a finite position and nonzero XYZW quaternion.")
    rotation /= np.linalg.norm(rotation)
    return wp.transform(wp.vec3(*position), wp.quat(*rotation))


def number(value, lower, upper, integer=False):
    value = float(value)
    if (
        not math.isfinite(value)
        or not lower <= value <= upper
        or integer
        and int(value) != value
    ):
        raise ValueError(
            f"Expected {'integer' if integer else 'number'} from {lower} to {upper}."
        )
    return int(value) if integer else value


def boxes(array):
    return [
        {
            "transform": row["transform"].tolist(),
            "half_extents": row["half_extents"].tolist(),
        }
        for row in array.numpy()
    ]


class Runtime:
    def __init__(self, message):
        self.rig = ModelRig(message.get("rig", "soma"))
        self.soma = self.rig is ModelRig.SOMA
        precision = message.get("precision", "fp32")
        if precision not in ("fp32", "fp16"):
            raise ValueError("Precision must be fp32 or fp16.")
        wp.init()
        wp.set_device("cuda:0")
        engine_type, fp32, fp16 = (
            (MotionBricksSOMAEngine, SOMA_N1_FP32_ONNX_PATH, SOMA_N1_FP16_ONNX_PATH)
            if self.soma
            else (
                MotionBricksUnitreeG1Engine,
                UNITREE_G1_N1_FP32_ONNX_PATH,
                UNITREE_G1_N1_FP16_ONNX_PATH,
            )
        )
        self.engine = engine_type(
            fp16 if precision == "fp16" else fp32,
            model_to_runtime=model_to_runtime(self.rig),
            build_options=TensorRTBuildOptions(enable_fp16=precision == "fp16"),
            cache_dir=ROOT / ".trt-cache",
        )
        self.precision = precision
        with wp.ScopedStream(self.engine.stream):
            self.assets = load_rig_assets(self.rig)
            args = (self.assets.model.skeleton,) if self.soma else ()
            self.context = self.engine.create_input(*args)
            self.target = self.engine.create_input(*args)
            self.output = self.engine.create_output(*args)
            self.sampled = wp.empty_like(self.context.poses)
            self.character = None
            self.instances = []
            self.record_file = None
            self.record_count = 0
            self.robot_state = None if self.soma else self.assets.model.state()
            self.trajectory_pose = wp.empty_like(self.assets.reference_pose)
        self.parents = (
            self.assets.model.skeleton.parent_indices.numpy().tolist()
            if self.soma
            else []
        )

    def pool(self):
        return SmartPrimitiveAssetPool(
            self.assets.clip_registry,
            self.assets.reference_pose,
            self.assets.model_to_runtime,
        )

    def describe(self):
        model = self.assets.model
        if self.soma:
            geometry = {
                "kind": "skeleton",
                "parents": self.parents,
                "names": model.skeleton.joint_names,
                "inverse_bind": model.inverse_bind_transforms.numpy().tolist(),
                "override_pose": model.override_poses["RelaxedHands"].numpy().tolist(),
                "override_mask": model.override_pose_masks["Fingers"].numpy().tolist(),
                "meshes": [
                    {
                        "positions": m.points.numpy().reshape(-1).tolist(),
                        "indices": m.indices.numpy().tolist(),
                        "joints": m.joint_indices.numpy().tolist(),
                        "weights": m.joint_weights.numpy().tolist(),
                        "influences": m.num_influences,
                    }
                    for m in model.skinned_meshes
                ],
            }
            from app.runtime.soma.presets import (
                DEFAULT_LOCOMOTION_MODES,
                STYLE_LOCOMOTION_MODES,
            )

            outlines = skeletal_mesh_sole_outlines(model, SOMA_N1_FOOT_CONTACT_JOINTS)
        else:
            meshes = []
            types, scales, colors, bodies, local = (
                a.numpy()
                for a in (
                    model.shape_type,
                    model.shape_scale,
                    model.shape_color,
                    model.shape_body,
                    model.shape_transform,
                )
            )
            for index in np.flatnonzero(
                model.shape_flags.numpy() & int(newton.ShapeFlags.VISIBLE)
            ):
                source = model.shape_source[index]
                item = {
                    "type": int(types[index]),
                    "scale": scales[index].tolist(),
                    "color": colors[index].tolist(),
                    "body": int(bodies[index]),
                    "transform": local[index].tolist(),
                }
                if source is not None and int(types[index]) in (8, 10):
                    item.update(
                        positions=np.asarray(source.vertices).reshape(-1).tolist(),
                        indices=np.asarray(source.indices).reshape(-1).tolist(),
                    )
                meshes.append(item)
            geometry = {
                "kind": "robot",
                "meshes": meshes,
                "names": list(model.body_label),
                "bones": [
                    [int(parent), int(child)]
                    for parent, child in zip(
                        model.joint_parent.numpy(), model.joint_child.numpy()
                    )
                    if parent >= 0 and child >= 0
                ],
            }
            from app.runtime.unitree_g1.presets import (
                DEFAULT_LOCOMOTION_MODES,
                STYLE_LOCOMOTION_MODES,
            )

            outlines = body_sole_outlines(
                model,
                UNITREE_G1_N1_FOOT_CONTACT_BODIES,
                UNITREE_G1_N1_FOOT_CONTACT_OFFSETS,
            )
        geometry["contacts"] = [
            {
                "contact": o.contact,
                "transform": o.transform,
                "points": o.points.tolist(),
            }
            for o in outlines
        ]
        pose = wp.clone(self.assets.reference_pose)
        if self.soma:
            host = pose.numpy()
            host[0] = list(wp.mul(self.assets.model_to_runtime, wp.transform(*host[0])))
            pose.assign(host)
        return {
            "loaded": True,
            "rig": self.rig.value,
            "precision": self.precision,
            "geometry": geometry,
            "pose": self.visual_pose(pose),
            "modes": [
                {"value": mode.name, "label": mode.label}
                for mode in DEFAULT_LOCOMOTION_MODES
            ],
            "styles": [mode.name for mode in STYLE_LOCOMOTION_MODES],
            "behaviors": list(behavior_types()),
        }

    def visual_pose(self, pose):
        if self.soma:
            return pose.numpy().tolist()
        newton.eval_fk(
            self.assets.model, pose, self.assets.model.joint_qd, self.robot_state
        )
        return self.robot_state.body_q.numpy().tolist()

    def visual_frames(self, poses):
        if self.soma:
            return poses.numpy().tolist()
        return [self.visual_pose(poses[index]) for index in range(poses.shape[0])]

    def sample(self, entry, anchor=0):
        clip = self.assets.clip_registry.load(resolve(entry["clip"])).animation
        seconds = number(
            entry.get("time", 0), 0, (clip.num_frames - 1) / clip.sample_rate
        )
        placement = transform(entry.get("transform", {}))
        clip.sample(
            seconds, self.sampled[anchor], root_offset=self.assets.model_to_runtime
        )
        root = self.sampled[anchor].numpy().reshape(-1)
        shift = wp.transform(
            wp.vec3(-float(root[0]), -float(root[1]), 0), wp.quat_identity()
        )
        offset = wp.mul(placement, wp.mul(shift, self.assets.model_to_runtime))
        for frame in range(4):
            clip.sample(
                seconds + (frame - anchor) / 30, self.sampled[frame], root_offset=offset
            )
        self.sample_offset, self.sample_seconds = offset, seconds
        return clip

    def document_path(self, message):
        if "document" in message:
            suffix = {
                "motionbricks.smart_primitive.authoring": ".smart_primitive.json",
                "motionbricks.scene": ".scene.json",
            }.get(message["document"].get("format"))
            if suffix is None:
                raise ValueError("Choose a native primitive or scene document.")
            name = "workspace/.preview/current" + suffix
            path = write_document(name, message["document"])
        else:
            path = resolve(message["path"])
        pending, seen = [path], set()
        # Check the complete reference graph before the native loader opens a clip.
        while pending:
            item = pending.pop()
            if item in seen:
                continue
            seen.add(item)
            data = read_document(virtual(item))

            def visit(ref):
                file = resolve(ref)
                if file.suffix == ".json":
                    pending.append(file)
                return ref

            refs(data, visit)
        return path

    def preview(self, message):
        path = self.document_path(message)
        return self.primitive_preview(path)

    def primitive_preview(self, path, placement=None):
        doc, _, _ = load_smart_primitive_document(
            self.assets.clip_registry,
            self.assets.reference_pose,
            self.assets.model_to_runtime,
            path,
        )
        placement = (
            wp.transform_identity()
            if placement is None
            else wp.mul(placement, wp.transform_inverse(doc.world_xform(doc.root)))
        )
        targets = []
        metadata = []
        # Authoring previews accept unfinished sequences; execution validates them separately.
        for node, target in doc.iter_target_poses():
            if target.clip:
                metadata.append(
                    {
                        "clip": virtual(target.clip.path),
                        "duration": target.duration,
                        "sample_rate": target.clip.animation.sample_rate,
                    }
                )
            if not doc.is_visible(node):
                continue
            world = wp.mul(placement, doc.target_pose_world_xform(node))
            raw = target.sampled_poses.numpy()
            roots = raw.reshape(4, -1)
            for root in roots:
                root[:7] = list(wp.mul(world, wp.transform(*root[:7])))
            self.sampled.assign(raw)
            trajectory = (
                [
                    np.asarray(wp.transform_point(world, wp.vec3(*row[:3]))).tolist()
                    for row in target.trajectory_transforms.numpy()
                ]
                if target.has_trajectory
                else []
            )
            targets.append(
                {
                    "id": node.id,
                    "name": node.name,
                    "poses": self.visual_frames(self.sampled[: target.skeleton_count]),
                    "show_mesh": target.show_mesh,
                    "show_skeletons": target.show_skeletons,
                    "show_velocity": target.show_velocity,
                    "show_trajectory": target.show_trajectory,
                    "trajectory": trajectory,
                    "color": list(target.color),
                }
            )
        outlines = [
            {
                "transform": list(wp.mul(placement, doc.world_xform(node))),
                "half_extents": [part.width / 2, part.depth / 2, part.height / 2],
            }
            for node, part in doc.iter_interaction_boxes()
            if doc.is_visible(node)
        ]
        sequences = [{"name": s.name, "targets": s.node_ids} for s in doc.sequences]
        return {
            "targets": targets,
            "clips": metadata,
            "boxes": outlines,
            "sequences": sequences,
            "valid": True,
        }

    def scene_preview(self, message):
        scene, _, _ = load_scene_document(self.document_path(message))
        if scene.rig != self.rig:
            raise ValueError("Scene and loaded rig differ.")
        check_reference_rigs(scene)
        scene = scene.flatten()
        result = {"targets": [], "boxes": [], "sequences": {}}
        for item in scene.smart_primitives:
            if item.reference is None:
                continue
            preview = self.primitive_preview(item.reference, item.transform)
            result["sequences"][item.id] = preview["sequences"]
            if item.show_target_poses:
                result["targets"].extend(
                    {**target, "owner": item.id, "color": list(item.target_pose_color)}
                    for target in preview["targets"]
                )
            if item.show_interaction_boxes:
                result["boxes"].extend(preview["boxes"])
        return result

    def generate(self, message):
        entries = message["entries"]
        if not 2 <= len(entries) <= 128:
            raise ValueError("Use a Start and 1–127 targets.")
        self.sample(entries[0], 3)
        self.context.update_from_poses(self.sampled, stream=self.engine.stream)
        frames, contacts, segments = [], [], []
        for index, entry in enumerate(entries[1:]):
            self.sample(entry)
            count = number(entry.get("num_poses", 4), 1, 4, True)
            low = number(entry.get("min_tokens", 8), 6, 16, True)
            high = number(entry.get("max_tokens", 14), low, 16, True)
            early = number(entry.get("early", 0), 0, 20, True)
            self.target.update_from_poses(
                self.sampled[:count], stream=self.engine.stream
            )
            self.engine.generate_animation(
                self.context,
                self.target,
                self.output,
                config=MotionBricksInferenceConfig(
                    allowed_token_counts=tuple(range(low, high + 1))
                ),
            )
            length = max(4, self.output.num_frames - early)
            raw = self.output.poses[:length].numpy()
            frames.append(raw)
            contacts.append(self.output.foot_contacts[:length].numpy())
            segments.append(
                {
                    "target": index + 1,
                    "frames": length,
                    "timing_ms": dict(self.engine.timer.milliseconds),
                }
            )
            # The next context is what actually played, including early transitions.
            self.context.update_from_poses(
                self.output.poses[length - 4 : length], stream=self.engine.stream
            )
        raw = np.concatenate(frames)
        contact = np.concatenate(contacts)
        return self.save_motion(raw, contact, segments)

    def save_motion(self, raw, contacts, segments=()):
        folder = WORK / "results" / uuid.uuid4().hex[:12]
        folder.mkdir(parents=True)
        np.savez_compressed(
            folder / "motion.npz",
            poses=raw,
            foot_contacts=contacts,
            sample_rate=30,
            rig=self.rig.value,
        )
        # Stream frames so a long recording never needs a second full pose copy.
        with (folder / "motion.json").open("w", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"rig": self.rig.value, "sample_rate": 30})[:-1]
                + ', "frames":['
            )
            for index, raw_pose in enumerate(raw):
                if index:
                    stream.write(",")
                self.trajectory_pose.assign(raw_pose)
                json.dump(
                    self.visual_pose(self.trajectory_pose), stream, allow_nan=False
                )
            stream.write('], "foot_contacts":[')
            for index, contact in enumerate(contacts):
                if index:
                    stream.write(",")
                json.dump(contact.tolist(), stream, allow_nan=False)
            stream.write('], "segments":')
            json.dump(segments, stream, allow_nan=False)
            stream.write("}")
        return {
            "motion": virtual(folder / "motion.json"),
            "archive": virtual(folder / "motion.npz"),
            "frames": len(raw),
            "segments": segments,
        }

    def scene(self, message):
        if self.record_file:
            raise ValueError("Finish recording before loading another scene.")
        path = self.document_path(message)
        scene, _, _ = load_scene_document(path)
        if scene.rig != self.rig:
            raise ValueError("Scene and loaded rig differ.")
        check_reference_rigs(scene)
        scene = scene.flatten()
        pool = self.pool()
        character_type = SOMACharacter if self.soma else UnitreeG1Character
        character = character_type(
            self.engine,
            asset_pool=pool,
            initial_transform=scene.character_start.transform,
        )
        instances, offset = [], 0
        allowed = behavior_types()
        for item in scene.smart_primitives:
            if item.reference is None:
                raise ValueError("Choose a primitive reference for " + item.name)
            if item.behavior not in allowed:
                raise ValueError("Select a native behavior for " + item.name)
            runtime = pool.instantiate(
                item.reference, smart_primitive_type=allowed[item.behavior]
            )
            runtime.set_node_world_transform(
                0, item.transform, stream=self.engine.stream
            )
            character.state_machine.interaction_state.prepare(character, runtime)
            instances.append((item, runtime, offset))
            offset += runtime.interaction_box_count
        self.character, self.instances = character, instances
        self.box_buffer = wp.empty(
            offset, dtype=InteractionBox, device=self.engine.device
        )
        self.overlaps = wp.empty(offset, dtype=wp.int32, device=self.engine.device)
        self.active = None
        self.blocked = set()
        self.blended = wp.empty_like(self.context.poses)
        self.blend_targets = wp.empty_like(self.context.poses)
        for _, runtime, offset in self.instances:
            runtime.get_world_interaction_boxes(
                self.box_buffer[offset : offset + runtime.interaction_box_count],
                self.engine.stream,
            )
        return {
            "scene_ready": True,
            "pose": self.visual_pose(character.current_pose),
            "targets": self.scene_targets(),
            "boxes": self.visible_boxes(),
        }

    def scene_targets(self):
        targets = []
        for item, runtime, _ in self.instances:
            if not item.show_target_poses:
                continue
            for index in runtime.asset.target_node_indices:
                runtime.get_world_target_poses(
                    int(index), self.sampled, self.engine.stream
                )
                targets.append(
                    {
                        "id": f"{item.id}:{index}",
                        "pose": self.visual_pose(self.sampled[0]),
                        "color": list(item.target_pose_color),
                    }
                )
        return targets

    def visible_boxes(self):
        all_boxes = boxes(self.box_buffer)
        return [
            box
            for item, runtime, offset in self.instances
            if item.show_interaction_boxes
            for box in all_boxes[offset : offset + runtime.interaction_box_count]
        ]

    def step(self, message):
        if self.character is None:
            raise ValueError("Open a scene first.")
        character = self.character
        facing = np.asarray(message.get("facing", [0, -1, 0]), dtype=np.float32)
        movement = np.asarray(message.get("movement", [0, 0, 0]), dtype=np.float32)
        if (
            facing.shape != (3,)
            or movement.shape != (3,)
            or not np.isfinite(np.r_[facing, movement]).all()
        ):
            raise ValueError("Movement and facing need finite XYZ vectors.")
        dt = (
            1 / 30 if self.record_file else number(message.get("dt", 1 / 30), 1e-6, 0.1)
        )
        character.step(
            dt,
            LocomotionCommand(
                facing, movement, bool(message.get("run")), message.get("mode")
            ),
        )
        for _, runtime, offset in self.instances:
            runtime.get_world_interaction_boxes(
                self.box_buffer[offset : offset + runtime.interaction_box_count],
                self.engine.stream,
            )
        touching = []
        if self.box_buffer.shape[0]:
            compute_interaction_box_overlaps(
                self.box_buffer, character.probe_box, self.overlaps, self.engine.stream
            )
            hits = self.overlaps.numpy()
            touching = [
                (item, runtime)
                for item, runtime, offset in self.instances
                if np.any(hits[offset : offset + runtime.interaction_box_count])
            ]
        machine = character.state_machine
        if self.active is not None and not machine.is_interacting:
            self.blocked.add(self.active)
            self.active = None
        self.blocked.intersection_update(item.id for item, _ in touching)
        if machine.is_interacting:
            if message.get("interact"):
                machine.interact()
        elif touching:
            item, runtime = min(
                touching,
                key=lambda pair: np.linalg.norm(
                    np.array(pair[0].transform.p) - character.character_ground_position
                ),
            )
            if (
                message.get("interact")
                or message.get("auto")
                and item.id not in self.blocked
            ):
                machine.start_interaction(character, runtime)
                self.active = item.id
        result = {
            "pose": self.visual_pose(character.current_pose),
            "foot_contacts": character.current_foot_contacts.numpy().tolist(),
            "position": character.character_ground_position.tolist(),
            "actual_velocity": character.actual_velocity.tolist(),
            "desired_velocity": character.movement.desired_velocity.tolist(),
            "input_velocity": character.movement.input_velocity.tolist(),
            "boxes": self.visible_boxes(),
            "probe": boxes(character.probe_box),
            "interacting": machine.is_interacting,
            "interactable": [item.id for item, _ in touching],
            "timing_ms": dict(self.engine.timer.milliseconds),
            "state": machine.state.name,
            "transition_seconds": character.time_to_transition(),
        }
        if message.get("scene_targets"):
            result["scene_targets"] = self.scene_targets()
        if message.get("debug"):
            character.player.refresh_context()
            result["context"] = self.visual_frames(character.player.context.poses)
            result["targets"] = self.visual_frames(
                character.player.target.poses[: character.player.target.num_poses]
            )
            if machine.is_locomoting:
                machine.active_locomotion_state.evaluate_current_target_poses(
                    character, self.blended, self.blend_targets
                )
                result["blendspace"] = self.visual_frames(self.blend_targets)
        if self.record_file:
            with self.record_file.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "pose": character.current_pose.numpy().tolist(),
                            "contacts": result["foot_contacts"],
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
            self.record_count += 1
        return result

    def record(self, message):
        if message.get("enabled"):
            if self.character is None or self.record_file:
                raise ValueError("Start a scene with no active recording first.")
            folder = WORK / "recordings"
            folder.mkdir(parents=True, exist_ok=True)
            self.record_file = folder / (uuid.uuid4().hex + ".jsonl")
            self.record_count = 0
            return {"recording": True}
        if not self.record_file or not self.record_count:
            self.record_file = None
            return {"recording": False}
        pose_path = self.record_file.with_suffix(".poses.npy")
        contact_path = self.record_file.with_suffix(".contacts.npy")
        raw = np.lib.format.open_memmap(
            pose_path,
            mode="w+",
            dtype=np.float32,
            shape=(self.record_count, *self.assets.reference_pose.numpy().shape),
        )
        contacts = np.lib.format.open_memmap(
            contact_path, mode="w+", dtype=np.float32, shape=(self.record_count, 4)
        )
        with self.record_file.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                raw[index], contacts[index] = row["pose"], row["contacts"]
        result = self.save_motion(raw, contacts)
        del raw, contacts
        pose_path.unlink()
        contact_path.unlink()
        self.record_file = None
        return {"recording": False, **result}

    def dispatch(self, message):
        with wp.ScopedStream(self.engine.stream):
            op = message.get("op")
            if op == "sample":
                clip = self.sample(message["entry"], 3 if message.get("start") else 0)
                trajectory = []
                scale = number(message["entry"].get("trajectory_scale", 1), 1, 4)
                for index in range(21):
                    clip.sample(
                        self.sample_seconds + (index - 10) * scale / 30,
                        self.trajectory_pose,
                        root_offset=self.sample_offset,
                    )
                    trajectory.append(
                        self.trajectory_pose.numpy().reshape(-1)[:3].tolist()
                    )
                return {
                    "poses": self.visual_frames(self.sampled),
                    "trajectory": trajectory,
                    "duration": (clip.num_frames - 1) / clip.sample_rate,
                    "sample_rate": clip.sample_rate,
                }
            operations = {
                "generate": self.generate,
                "preview": self.preview,
                "scene_preview": self.scene_preview,
                "scene": self.scene,
                "step": self.step,
                "record": self.record,
            }
            if op not in operations:
                raise ValueError("Unknown native operation.")
            return operations[op](message)
