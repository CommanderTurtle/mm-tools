# Scene Reconstruction Benchmarks

This package evaluates Fire3D reconstruction with ground-truth instance IDs
and poses. It reports object geometry and foreground-only PBR appearance,
separately from perception quality.

The checked-in `ithor_v1.json` and `imaginarium_v1.json` manifests address
objects by stable scene and instance IDs. Download the matching data first:

```bash
fire3d download --data --dataset ithor
fire3d download --data --dataset imaginarium
```

Validate the data/manifest contract:

```bash
python -m benchmarks.scene_reconstruction.validate_dataset \
  --manifest benchmarks/scene_reconstruction/manifests/ithor_v1.json
```

Generate reconstruction results from ground-truth masks and poses:

```bash
python -m benchmarks.scene_reconstruction.run_lc64_geometry \
  --manifest benchmarks/scene_reconstruction/manifests/ithor_v1.json \
  --scene-id iTHOR_FloorPlan312_physics \
  --output-root results/oracle_reconstruction \
  --inference-num-steps 12 \
  --object-batch-size 16 \
  --predict-appearance
```

Evaluate geometry and appearance:

```bash
python -m benchmarks.scene_reconstruction.evaluate_geometry \
  --manifest benchmarks/scene_reconstruction/manifests/ithor_v1.json \
  --prediction-root results/oracle_reconstruction \
  --prediction-space ff_canonical \
  --output results/oracle_reconstruction/geometry.json

python -m benchmarks.scene_reconstruction.evaluate_appearance \
  --manifest benchmarks/scene_reconstruction/manifests/ithor_v1.json \
  --prediction-root results/oracle_reconstruction \
  --output results/oracle_reconstruction/appearance
```

Geometry uses symmetric Chamfer-L1, F1, precision, recall, and normal
consistency in world coordinates. Object means are computed within each scene,
then scenes are weighted equally. Appearance uses the most visible source
camera, GT instance pixels, and the canonical PBR Blender recipe; it reports
foreground PSNR together with crop/full-image diagnostics and mask coverage.

For a protocol smoke test without model inference, pass
`--identity-gt-prediction --max-scenes 1` to `evaluate_geometry`.

ShapeR evaluation data is separate:

```bash
fire3d download --evaluation shaper
python -m baselines.shaper.eval_shaper_reconstruction_geometry --help
```
