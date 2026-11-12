# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Export the Python inference models from the released checkpoints."""

import argparse
import gc
import importlib.metadata
import logging
import os
from pathlib import Path
import sys
import tempfile
import warnings

import onnx
import torch

from .checkpoints import checkpoint_root, load_sources
from .checksums import CHECKPOINTS

logger = logging.getLogger(__name__)


def check_environment(kind):
    """Check that Python and the key packages match the pinned export environment."""
    python = (3, 12)
    versions = {
        "torch": "2.14.0",
        "onnx": "1.23.2",
        "numpy": "1.26.4",
        "safetensors": "0.8.0",
        "protobuf": "7.36.1",
    }
    if sys.version_info[:2] != python:
        raise ValueError(
            f"The exporter requires Python {python[0]}.{python[1]}. Use the command in tools/onnx/README.md."
        )
    for package, expected in versions.items():
        actual = importlib.metadata.version(package).split("+")[0]
        if actual != expected:
            raise ValueError(
                f"Expected {package}=={expected}, found {actual}. Use the locked exporter environment."
            )


def example_inputs(kind):
    """Create trace inputs: a batch of two requests with different allowed token counts."""
    allowed = torch.eye(11, dtype=torch.int32)[[0, 10]]
    masks = [torch.ones(2, 8, dtype=torch.int32) for _ in range(3)]
    for mask in masks:
        mask[:, 4:6] = 0
    if kind == "g1":
        context = torch.zeros(2, 4, 36)
        context[..., 2] = 0.75
        context[..., 6] = 1.0  # Identity quaternion, XYZW.
        target = context.clone()
        target[..., 0] = 1.0
        return (context, target, *masks, allowed)

    global_root = torch.zeros(2, 8, 5)
    global_root[..., 1] = 1.0
    global_root[..., 3] = 1.0  # cos(heading), with heading zero.
    local_root = torch.zeros(2, 8, 4)
    local_root[..., 3] = 1.0
    local_pose = torch.zeros(2, 8, 267)
    # Each joint's 6D rotation contains the first two columns of identity.
    rotations = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).repeat(30)
    local_pose[..., 87:] = rotations
    return (global_root, masks[0], local_root, masks[1], local_pose, masks[2], allowed)


def export_fp32(kind, model, output):
    """Trace the batched inference graph with the pinned export options."""
    if kind == "g1":
        from .models.g1 import INPUT_NAMES, OUTPUT_NAMES
    else:
        from .models.soma import INPUT_NAMES, OUTPUT_NAMES
    kwargs = dict(
        input_names=list(INPUT_NAMES),
        output_names=list(OUTPUT_NAMES),
        dynamic_axes={name: {0: "batch_size"} for name in INPUT_NAMES + OUTPUT_NAMES},
        opset_version=17,
        dynamo=False,
        do_constant_folding=(kind == "soma"),
    )
    with warnings.catch_warnings():
        # The shape and configuration checks are specific to these release architectures.
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        warnings.filterwarnings(
            "ignore", category=FutureWarning, module=r"torch\.onnx\..*"
        )
        if kind == "g1":
            torch.backends.mha.set_fastpath_enabled(False)
            with torch.inference_mode():
                torch.onnx.export(model, example_inputs(kind), str(output), **kwargs)
        else:
            torch.onnx.export(model, example_inputs(kind), str(output), **kwargs)


def validate_output(path, kind):
    """Check graph validity and the batched interface used by the demos."""
    try:
        onnx.checker.check_model(str(path), full_check=True)
    except (onnx.checker.ValidationError, onnx.shape_inference.InferenceError) as error:
        raise ValueError(f"Invalid ONNX model: {error}") from error

    model = onnx.load(path)
    if kind == "g1":
        from .models.g1 import INPUT_NAMES

        outputs = {
            "pred_qpos": (onnx.TensorProto.FLOAT, (64, 36)),
            "pred_num_tokens": (onnx.TensorProto.INT32, ()),
            "pred_foot_contacts": (onnx.TensorProto.FLOAT, (64, 4)),
        }
    else:
        from .models.soma import INPUT_NAMES

        outputs = {
            "pred_global_motions": (onnx.TensorProto.FLOAT, (64, 366)),
            "pred_num_tokens": (onnx.TensorProto.INT32, ()),
        }
    inputs = {
        name: (
            onnx.helper.np_dtype_to_tensor_dtype(value.numpy().dtype),
            value.shape[1:],
        )
        for name, value in zip(INPUT_NAMES, example_inputs(kind), strict=True)
    }
    batch_dimension = None
    for mode, values, expected in (
        ("input", model.graph.input, inputs),
        ("output", model.graph.output, outputs),
    ):
        if {value.name for value in values} != set(expected):
            raise ValueError(
                f"Unexpected ONNX {mode} names. Expected {list(expected)}."
            )
        for value in values:
            dtype, shape = expected[value.name]
            tensor = value.type.tensor_type
            dimensions = tensor.shape.dim
            if tensor.elem_type != dtype or len(dimensions) != len(shape) + 1:
                raise ValueError(f"Unexpected ONNX dtype or rank for {value.name}.")
            batch = dimensions[0].dim_param
            if not batch or (batch_dimension is not None and batch != batch_dimension):
                raise ValueError(
                    f"{value.name} must share the symbolic batch dimension."
                )
            batch_dimension = batch
            for dimension, size in zip(dimensions[1:], shape, strict=True):
                # The TorchScript exporter leaves some SOMA output sizes symbolic.
                # TensorRT resolves and checks them when it builds the engine.
                if mode == "output" and dimension.dim_param:
                    continue
                if not dimension.HasField("dim_value") or dimension.dim_value != size:
                    raise ValueError(f"Unexpected ONNX shape for {value.name}.")


def export_release(variant, checkpoints, output, force=False):
    """Verify the checkpoints, then export and validate the ONNX model before saving it."""
    kind, separator, precision = variant.partition("_")
    if not separator or kind not in CHECKPOINTS or precision not in ("fp32", "fp16"):
        raise ValueError(f"Unknown model variant: {variant}")
    output = Path(output).expanduser().resolve()
    if output.exists() and not force:
        raise ValueError(f"Output already exists: {output}. Use --force to replace it.")
    check_environment(kind)
    torch.set_num_threads(1)
    sources = load_sources(checkpoint_root(Path(checkpoints), CHECKPOINTS[kind]), CHECKPOINTS[kind])
    if kind == "g1":
        from .models.g1 import build_g1

        model = build_g1(sources)
    else:
        from .models.soma import build_soma

        model = build_soma(sources)
    # load_state_dict copies the weights, so release the loaded checkpoint before exporting.
    del sources
    gc.collect()

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="motionbricks-export-", dir=output.parent
    ) as temporary:
        staged = Path(temporary) / "model.onnx"
        logger.debug("Exporting the batched %s model...", kind.upper())
        export_fp32(kind, model, staged)
        del model
        gc.collect()
        if precision == "fp16":
            logger.debug("Converting to mixed FP16...")
            graph = onnx.load(staged)
            if kind == "g1":
                from .precision import g1_fp16

                graph = g1_fp16(graph)
            else:
                from .precision import soma_fp16

                graph = soma_fp16(graph)
            onnx.save(graph, staged)
            del graph
        validate_output(staged, kind)
        if force:
            staged.replace(output)
        else:
            # A hard link fails if another process created the output meanwhile.
            os.link(staged, output)
    logger.info("Saved %s. The ONNX graph and tensor interface checks passed.", output)
    return {
        "variant": variant,
        "bytes": output.stat().st_size,
    }


class _MessageFormatter(logging.Formatter):
    """Show messages as they are, with the level in front of warnings and errors."""

    def format(self, record):
        message = super().format(record)
        return f"{record.levelname}: {message}" if record.levelno >= logging.WARNING else message


def configure_logging(args):
    """Show the result and any problems, or every step with --verbose."""
    handler = logging.StreamHandler()
    handler.setFormatter(_MessageFormatter())
    logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True)
    level = logging.DEBUG if args.verbose else logging.INFO
    for name in ("__main__", __package__):
        logging.getLogger(name).setLevel(level)


def main():
    """Parse the command line and export one model."""
    parser = argparse.ArgumentParser(
        description="Export SOMA or Unitree G1 checkpoints to batched ONNX."
    )
    parser.add_argument("--model", choices=("soma", "unitree_g1"), required=True, help="Model to export.")
    parser.add_argument(
        "--checkpoints",
        required=True,
        type=Path,
        help="Downloaded checkpoint folder, such as checkpoints/MotionBricks/soma.",
    )
    parser.add_argument(
        "--precision", choices=("fp32", "fp16"), default="fp32", help="fp32, or fp16 for mixed FP16."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (default: mmm_v6_{soma|g1}_{fp32|fp16}.onnx).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output after all checks pass.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show each export step and verified checkpoint file.")
    args = parser.parse_args()
    configure_logging(args)
    kind = {"soma": "soma", "unitree_g1": "g1"}[args.model]
    variant = f"{kind}_{args.precision}"
    output = args.output or Path(f"mmm_v6_{variant}.onnx")
    try:
        export_release(variant, args.checkpoints, output, force=args.force)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Export failed: {error}\n")


if __name__ == "__main__":
    main()
