# MotionBricks studio

SOMA and Unitree G1 motion authoring, backed by MotionBricks
`54fc563aeaaf819a753d72cc9fdc6713989cdd43`.

Select `motionbricks` in `models/download_models.py`, then:

```bash
./setupwithuv.sh
./startwithuv.sh
```

Open `http://127.0.0.1:8221`. Setup uses Linux, an NVIDIA CUDA-capable driver,
uv and Bun; the lane's Python 3.12 project metadata resolves every dependency.
`HOST` and `PORT` override the local listener; keep access on a trusted host.

Load a rig when needed. FP32 is the default; mixed FP16 is an explicit choice.
First use builds a local TensorRT plan. Unload/cancel releases the native process;
closing the browser or server also releases it. Workspace files remain on disk.

| Workspace | Controls |
| --- | --- |
| Inbetweener | Start/target clips, pose windows, transforms, token ranges, early transitions, trajectories, playback and scrubbing |
| Smart Primitive | Hierarchy, transform locks, target poses, SOMA IK pins, interaction boxes, constrained pivots and ordered sequences |
| Scene editor | Native primitive references and behaviors, parented placements, cubes, character start and viewport overlays |
| Scene preview | Camera-relative movement, native locomotion styles, manual/automatic interactions, context/blendspace inspection and recording |

Orbit with the mouse; Ctrl+wheel changes field of view. W/E switch edit gizmos,
F frames the scene. Ctrl+Z / Ctrl+Shift+Z undo/redo; Ctrl+S saves. In Scene preview,
WASD/arrows move, Space runs, I interacts, O/P toggle strafe/crouch and brackets
cycle styles. Standard gamepads use the left stick for movement/speed, right
stick for orbit, X for interaction, Y/B for strafe/crouch and A to cycle styles.

Open native documents from the library, or import a folder to preserve its
relative clip/primitive references. Save writes portable JSON to `workspace/`;
Export project includes the referenced clips and primitives in a ZIP. Generated
motion and recordings provide native pose/contact arrays in NPZ and browser
playback JSON. Recording frames are also written incrementally to JSONL.

Models and samples come from the shared downloader cache; `models/which-ones.txt`
maps each bundle to its files and destination. The SOMA and Unitree G1 runtime
code itself is tracked beside the studio in `motionbricks/` and `app/`.
