# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

try:
    import tensorrt as trt
    from polygraphy.backend.trt import CreateConfig, Profile, engine_from_network, network_from_onnx_path, save_engine

    HAVE_TRT = True

except (ImportError, ModuleNotFoundError):

    HAVE_TRT = False


def build_engine(
    onnx_path,
    output_path,
    fp16,
    input_profile=None,
    enable_refit=False,
    enable_preview=False,
    timing_cache=None,
    workspace_size=0,
    int8=False,
    builder_optimization_level=None,
):
    """Build and serialize a TensorRT engine from an ONNX model."""
    print(f"Building TensorRT engine for {onnx_path}: {output_path}")
    p = Profile()
    if input_profile:
        for name, dims in input_profile.items():
            assert len(dims) == 3
            p.add(name, min=dims[0], opt=dims[1], max=dims[2])

    preview_features = None

    config_kwargs = {}
    if workspace_size > 0:
        config_kwargs["memory_pool_limits"] = {trt.MemoryPoolType.WORKSPACE: workspace_size}
    engine = engine_from_network(
        network_from_onnx_path(onnx_path),
        config=CreateConfig(
            fp16=fp16,
            refittable=enable_refit,
            profiles=[p],
            preview_features=preview_features,
            load_timing_cache=timing_cache,
            int8=int8,
            builder_optimization_level=builder_optimization_level,
            **config_kwargs,
        ),
        save_timing_cache=timing_cache,
    )
    save_engine(engine, path=output_path)
