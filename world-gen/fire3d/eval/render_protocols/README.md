# Unified Render Protocols

Render protocols are immutable presentation contracts for
`eval/unified_render.py`. They are intentionally separate from
`configs/inference/`, whose files define perception, reconstruction, camera, and
evaluation behavior together.

Use an inference protocol and a render protocol independently:

```bash
python eval/unified_render.py \
  --dataset imaginarium \
  --source comparison \
  --run-root results/<inference-run>/imaginarium \
  --scene-id bedroom_01 \
  --config results/<camera-run>/views.yaml \
  --protocol inference_sep4_v5 \
  --render-protocol render_sep5_v0 \
  --output-root results/<render-run>
```

`render_sep5_v0` freezes the plain September 5 baseline before presentation
experiments. It records both task-level values and effective settings that had
previously lived only in Blender Python defaults: raster engine, color
management, lighting, camera clipping, output encoding, imported-material
policy, normals/smoothing, geometry material, and background defaults.

Do not edit a frozen protocol after it has produced referenced images. Create
`render_sep5_v1`, `render_sep5_v2`, and so on, changing one controlled factor
at a time. Prefer style parameters already consumed by
`scene_comparison_blender.apply_beauty_style`, such as texture interpolation,
auto smooth, weighted normals, roughness floor, metallic scale, light energy,
and color grade. A new unsupported engine or material operation requires code
and a validation render, not an undocumented JSON key.

Candidate protocols may declare `"extends": "render_sep5_v0"`. The loader
recursively deep-merges object fields while replacing scalar and list fields;
it rejects inheritance cycles and records a hash of the fully resolved
contract. This keeps one authoritative copy of unchanged raster, camera,
background, and output settings.

The first eight-view candidate round uses texture-only protocols:

| Protocol | Controlled change |
| --- | --- |
| `render_sep5_v1_normals` | 40-degree auto smooth plus weighted corner normals |
| `render_sep5_v2_cubic` | Cubic interpolation for imported texture nodes |
| `render_sep5_v3_material` | Metallic scale 0.2, roughness floor 0.35, specular 0.2 |
| `render_sep5_v4_lighting` | Reduced camera Sun, soft area lights, ray tracing, 64 samples |
| `render_sep5_v5_balanced` | Conservative combination of v1 through v4 |
| `render_sep5_v6_surface` | Successful v1 normals plus v2 filtering, with baseline light/materials |

These candidates intentionally exclude geometry smoothing, texture-hole
repair, dark lifting, and color grading. Their fixed cameras live under
`eval/render_view_sets/sep5_imaginarium_manual8/`.
