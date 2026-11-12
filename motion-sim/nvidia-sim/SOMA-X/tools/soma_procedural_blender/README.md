# SOMA Procedural Blender Reference

This directory contains a reference Blender add-on for consuming
`assets/SOMA_procedural_transforms.json`. Blender does not have a Maya-style
dependency node that can directly own armature output plugs, so the reference
uses Blender's native add-on pattern:

1. Configure one neutral armature object with the JSON sidecar path.
2. Capture the imported twist pose-bone baselines.
3. Read public SOMA pose-bone local transforms in the sidecar order.
4. Evaluate the same dense NumPy parameter matrices used by the Maya reference.
5. Write generated transform matrices onto existing procedural twist pose bones.

The handler reconstructs each public source local transform from the posed
armature hierarchy for twist extraction and uses armature-space pose-bone
matrices for translation interpolation. That is the Blender armature equivalent
of the Maya node's `inputLocalMatrix[]` and `inputWorldMatrix[]` contract while
preserving the USD importer's rest/baseline bone orientations. Generated twist
rotations are inserted between the parent rest orientation and the twist-bone
rest orientation, matching the Maya setup helper.

## Install

Blender is already installed on this machine. From the repository root:

```bash
.venv/bin/python tools/soma_procedural_blender/setup_blender_addon.py --force
```

The setup script installs `soma_procedural_blender` into the current user's
Blender `scripts/addons` directory. On Linux and macOS it creates a development
symlink by default. Use `--copy` if you need a physical copy.

Use `--dry-run` to inspect the target path without writing.

## Use In Blender

1. Enable the add-on named `SOMA Procedural Transforms`.
2. Select the neutral armature that already contains the public SOMA bones and
   procedural twist bones named in `assets/SOMA_procedural_transforms.json`.
3. Open `Properties > Object > SOMA Procedural`.
4. Set the definition path if auto-discovery does not find the repository copy.
5. Click `Configure`.

Configure before posing or animating the armature, because configuration stores
the imported neutral twist pose as the output baseline. After configuration,
Blender's depsgraph and frame-change handlers evaluate the armature when it
updates.
`Update Now` runs a single immediate evaluation.

## Runtime Contract

The armature must already contain:

- all main pose bones from `public_rig_derivation.main_joint_names`
- all generated twist pose bones from every segment `twist_joints` entry

The add-on writes metadata custom properties to the armature and to each twist
pose bone. The important armature properties are:

- `soma_procedural_definition_path`
- `soma_procedural_enabled`
- `soma_procedural_public_joint_names`
- `soma_procedural_twist_joint_names`

Typical evaluation logic:

```python
from soma_procedural_blender.blender_reference import configure_armature, evaluate_armature

configure_armature(
    bpy.context.object,
    definition_path="/path/to/SOMA-X/assets/SOMA_procedural_transforms.json",
)
evaluate_armature(bpy.context.object)
```

## Reference Scope

This is a host reference for the portable sidecar, not a production Blender rig
system. It exercises name resolution, sidecar validation, NumPy matrix loading,
and live armature evaluation without depending on the SOMA Python runtime or
Torch inside Blender.
