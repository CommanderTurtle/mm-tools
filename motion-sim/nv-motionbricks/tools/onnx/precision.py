# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Mixed precision conversion for the released ONNX models."""

from collections import defaultdict
import numpy as np
import onnx
from onnx import TensorProto as T, helper, numpy_helper


def g1_fp16(model):
    """Use FP16 for the neural network and FP32 for sensitive and geometric operations."""
    # matrix_quat gathers exactly one candidate at axis -2: (N,64,1,4).
    # Inline the exporter's guaranteed singleton Squeeze branch.
    conditionals = [n for n in model.graph.node if n.op_type == "If"]
    assert len(conditionals) == 1 and conditionals[0].name == "/post/If"
    conditional = conditionals[0]
    branch = next(a.g for a in conditional.attribute if a.name == "then_branch")
    assert [n.op_type for n in branch.node] == ["Constant", "Squeeze"]
    assert numpy_helper.to_array(branch.node[0].attribute[0].t).tolist() == [-2]
    branch.node[-1].output[0] = conditional.output[0]
    nodes = []
    for node in model.graph.node:
        nodes.extend(branch.node if node.name == conditional.name else [node])
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return convert_networks_to_fp16(model, ("/core/backbone_net/", "/core/decoder/"))


def soma_fp16(model):
    """Convert the SOMA predictors and decoder, keeping motion geometry in FP32."""
    return convert_networks_to_fp16(model, ("/mmm_net/", "/decoder/"))


def convert_networks_to_fp16(model, prefixes):
    """Convert network operations and insert casts at precision boundaries."""
    if any(n.op_type in {"If", "Loop", "Scan"} for n in model.graph.node):
        raise ValueError(
            "Mixed precision conversion requires a graph without control flow."
        )
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    graph = model.graph
    values = list(graph.input) + list(graph.value_info) + list(graph.output)
    original = {v.name: v.type.tensor_type.elem_type for v in values}
    original.update({v.name: v.data_type for v in graph.initializer})
    assert all(name in original for n in graph.node for name in n.output)
    # Keep the numerically sensitive normalization and attention reductions in full
    # precision. Geometry, integration, and the public I/O stay in full precision too.
    precision = {
        n.name: (
            T.FLOAT16
            if n.name.startswith(prefixes)
            and n.op_type not in ("LayerNormalization", "Softmax", "Resize")
            else T.FLOAT
        )
        for n in graph.node
    }
    consumers = defaultdict(set)
    for n in graph.node:
        for name in n.input:
            consumers[name].add(precision[n.name])
    actual = dict(original)
    for tensor in graph.initializer:
        if tensor.data_type == T.FLOAT and consumers[tensor.name] == {T.FLOAT16}:
            tensor.CopyFrom(
                numpy_helper.from_array(
                    numpy_helper.to_array(tensor).astype(np.float16), tensor.name
                )
            )
            actual[tensor.name] = T.FLOAT16
    for n in graph.node:
        for name in n.output:
            if original[name] == T.FLOAT:
                actual[name] = precision[n.name]
    rewritten, cast_names = [], {}

    def cast_input(name, dtype):
        key = (name, dtype)
        if key not in cast_names:
            out = name + ("__fp16" if dtype == T.FLOAT16 else "__fp32")
            assert out not in actual
            rewritten.append(
                helper.make_node("Cast", [name], [out], name=out + "_cast", to=dtype)
            )
            cast_names[key] = out
            actual[out] = dtype
        return cast_names[key]

    for n in graph.node:
        dtype = precision[n.name]
        for i, name in enumerate(n.input):
            if original.get(name) == T.FLOAT and actual[name] != dtype:
                n.input[i] = cast_input(name, dtype)
        if dtype == T.FLOAT16:
            for attr in n.attribute:
                if (
                    attr.type == onnx.AttributeProto.TENSOR
                    and attr.t.data_type == T.FLOAT
                ):
                    # Ordinary IEEE half conversion, which keeps the attention -inf values.
                    attr.t.CopyFrom(
                        numpy_helper.from_array(
                            numpy_helper.to_array(attr.t).astype(np.float16),
                            attr.t.name,
                        )
                    )
                elif n.op_type == "Cast" and attr.name == "to" and attr.i == T.FLOAT:
                    attr.i = T.FLOAT16
        rewritten.append(n)
    for v in graph.value_info:
        if original[v.name] == T.FLOAT:
            v.type.tensor_type.elem_type = actual[v.name]
    assert all(actual[v.name] == v.type.tensor_type.elem_type for v in graph.output)
    del graph.node[:]
    graph.node.extend(rewritten)
    # Re-infer all new boundary edges and check operator dtype consistency.
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model
