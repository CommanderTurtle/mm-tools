# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Maya Python plug-in for evaluating SOMA procedural-control definitions.

Load in Maya after running ``setup_maya_plugin.py``:

    loadPlugin soma_procedural_maya_plugin.py
    somaCreateProceduralRigReference -definitionPath "/path/to/SOMA_procedural_transforms.json"
"""

import sys
from pathlib import Path

import maya.api.OpenMaya as om
import numpy as np


def _plugin_root() -> Path:
    try:
        plugin_file = Path(__file__)
    except NameError:
        plugin_file = Path(sys._getframe().f_code.co_filename)
    return plugin_file.resolve().parents[1]


PLUGIN_ROOT = _plugin_root()
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from soma_procedural_maya.evaluator import SomaProceduralTransformEvaluator  # noqa: E402
from soma_procedural_maya.maya_reference import (  # noqa: E402
    create_reference_scene_nodes,
    find_repo_definition,
    setup_template_rig_scene,
)

COMMAND_NAME = "somaCreateProceduralRigReference"
NODE_TYPE_NAME = "somaProceduralTransforms"
NODE_TYPE_ID = om.MTypeId(0x0007BEEF)


def maya_useNewAPI():
    """Tell Maya to pass Python API 2.0 objects into plug-in entrypoints."""


def _matrix_to_numpy(matrix):
    maya_matrix = np.array(
        [[matrix.getElement(row, column) for column in range(4)] for row in range(4)],
        dtype=np.float64,
    )
    # Maya transform matrices store translation in the last row. The portable
    # evaluator uses the column-vector convention with translation in [:3, 3].
    return maya_matrix.T


def _numpy_to_matrix(matrix):
    maya_matrix = np.asarray(matrix, dtype=np.float64).T
    return om.MMatrix([float(maya_matrix[row, column]) for row in range(4) for column in range(4)])


class SomaProceduralTransformsNode(om.MPxNode):
    """Evaluate procedural twist-joint transforms from public SOMA joint matrices."""

    definition_path = om.MObject()
    input_local_matrix = om.MObject()
    input_world_matrix = om.MObject()
    input_bind_world_matrix = om.MObject()
    output_transform = om.MObject()

    def __init__(self):
        super().__init__()
        self._cache_key = None
        self._evaluator = None

    @staticmethod
    def creator():
        return SomaProceduralTransformsNode()

    @staticmethod
    def initialize():
        typed_attr = om.MFnTypedAttribute()
        SomaProceduralTransformsNode.definition_path = typed_attr.create(
            "definitionPath",
            "dfp",
            om.MFnData.kString,
        )
        typed_attr.storable = True
        typed_attr.usedAsFilename = True

        matrix_attr = om.MFnMatrixAttribute()
        SomaProceduralTransformsNode.input_local_matrix = matrix_attr.create(
            "inputLocalMatrix",
            "ilm",
            om.MFnMatrixAttribute.kDouble,
        )
        matrix_attr.array = True
        matrix_attr.usesArrayDataBuilder = True
        matrix_attr.storable = False

        matrix_attr = om.MFnMatrixAttribute()
        SomaProceduralTransformsNode.input_world_matrix = matrix_attr.create(
            "inputWorldMatrix",
            "iwm",
            om.MFnMatrixAttribute.kDouble,
        )
        matrix_attr.array = True
        matrix_attr.usesArrayDataBuilder = True
        matrix_attr.storable = False

        matrix_attr = om.MFnMatrixAttribute()
        SomaProceduralTransformsNode.input_bind_world_matrix = matrix_attr.create(
            "inputBindWorldMatrix",
            "ibwm",
            om.MFnMatrixAttribute.kDouble,
        )
        matrix_attr.array = True
        matrix_attr.usesArrayDataBuilder = True
        matrix_attr.storable = False

        matrix_attr = om.MFnMatrixAttribute()
        SomaProceduralTransformsNode.output_transform = matrix_attr.create(
            "outputTransform",
            "otm",
            om.MFnMatrixAttribute.kDouble,
        )
        matrix_attr.array = True
        matrix_attr.usesArrayDataBuilder = True
        matrix_attr.writable = False
        matrix_attr.storable = False

        node = SomaProceduralTransformsNode
        node.addAttribute(node.definition_path)
        node.addAttribute(node.input_local_matrix)
        node.addAttribute(node.input_world_matrix)
        node.addAttribute(node.input_bind_world_matrix)
        node.addAttribute(node.output_transform)
        node.attributeAffects(node.definition_path, node.output_transform)
        node.attributeAffects(node.input_local_matrix, node.output_transform)
        node.attributeAffects(node.input_world_matrix, node.output_transform)
        node.attributeAffects(node.input_bind_world_matrix, node.output_transform)

    def _evaluator_for_data(self, data):
        raw_path = data.inputValue(self.definition_path).asString()
        if raw_path:
            definition_path = raw_path
        else:
            found = find_repo_definition()
            if found is None:
                raise FileNotFoundError(
                    "Could not find assets/SOMA_procedural_transforms.json. "
                    "Set definitionPath on the node."
                )
            definition_path = str(found)
        cache_key = definition_path
        if cache_key != self._cache_key:
            self._evaluator = SomaProceduralTransformEvaluator.from_path(definition_path)
            self._cache_key = cache_key
        return self._evaluator

    def _read_matrix_array(self, data, attribute, expected_count):
        matrices = np.broadcast_to(np.eye(4, dtype=np.float64), (expected_count, 4, 4)).copy()
        array_handle = data.inputArrayValue(attribute)
        for physical_index in range(len(array_handle)):
            array_handle.jumpToPhysicalElement(physical_index)
            logical_index = array_handle.elementLogicalIndex()
            if 0 <= logical_index < expected_count:
                matrices[logical_index] = _matrix_to_numpy(array_handle.inputValue().asMatrix())
        return matrices

    def compute(self, plug, data):
        if plug.attribute() != self.output_transform:
            return None

        evaluator = self._evaluator_for_data(data)
        local_matrices = self._read_matrix_array(
            data,
            self.input_local_matrix,
            len(evaluator.public_joint_names),
        )
        world_matrices = self._read_matrix_array(
            data,
            self.input_world_matrix,
            len(evaluator.public_joint_names),
        )
        bind_world_matrices = self._read_matrix_array(
            data,
            self.input_bind_world_matrix,
            len(evaluator.public_joint_names),
        )
        evaluation = evaluator.evaluate(
            local_matrices,
            source_world_matrices=world_matrices,
            source_bind_world_matrices=bind_world_matrices,
        )

        output_array = data.outputArrayValue(self.output_transform)
        builder = output_array.builder()
        for output_index in range(len(evaluation.joint_names)):
            output_handle = builder.addElement(output_index)
            output_handle.setMMatrix(_numpy_to_matrix(evaluation.matrices[output_index]))
        output_array.set(builder)
        output_array.setAllClean()
        return None


class SomaCreateProceduralRigReferenceCommand(om.MPxCommand):
    """Create a SOMA procedural node plus optional locator outputs."""

    @staticmethod
    def creator():
        return SomaCreateProceduralRigReferenceCommand()

    @staticmethod
    def syntax_creator():
        syntax = om.MSyntax()
        syntax.addFlag("-dp", "-definitionPath", om.MSyntax.kString)
        syntax.addFlag("-nn", "-nodeName", om.MSyntax.kString)
        syntax.addFlag("-nl", "-noLocators")
        syntax.addFlag("-str", "-setupTemplateRig")
        syntax.addFlag("-tap", "-templateAssetPath", om.MSyntax.kString)
        syntax.addFlag("-rr", "-rigRoot", om.MSyntax.kString)
        return syntax

    def doIt(self, args):
        arg_data = om.MArgDatabase(self.syntax(), args)
        definition_path = None
        node_name = "SOMA_proceduralTransforms"
        create_locators = True
        setup_template_rig = False
        template_asset_path = None
        rig_root = "SOMA_templateRig_PROC_GRP"

        if arg_data.isFlagSet("-definitionPath"):
            definition_path = arg_data.flagArgumentString("-definitionPath", 0)
        if arg_data.isFlagSet("-nodeName"):
            node_name = arg_data.flagArgumentString("-nodeName", 0)
        if arg_data.isFlagSet("-noLocators"):
            create_locators = False
        if arg_data.isFlagSet("-setupTemplateRig"):
            setup_template_rig = True
        if arg_data.isFlagSet("-templateAssetPath"):
            setup_template_rig = True
            template_asset_path = arg_data.flagArgumentString("-templateAssetPath", 0)
        if arg_data.isFlagSet("-rigRoot"):
            rig_root = arg_data.flagArgumentString("-rigRoot", 0)

        if setup_template_rig:
            summary = setup_template_rig_scene(
                definition_path=definition_path,
                template_asset_path=template_asset_path,
                rig_root=rig_root,
                node_name=node_name,
            )
            self.setResult(summary["procedural_node"])
            return

        node = create_reference_scene_nodes(
            definition_path=definition_path,
            node_name=node_name,
            create_locators=create_locators,
        )
        self.setResult(node)


def initializePlugin(plugin):
    plugin_fn = om.MFnPlugin(plugin, "NVIDIA", "0.1.0", "Any")
    plugin_fn.registerNode(
        NODE_TYPE_NAME,
        NODE_TYPE_ID,
        SomaProceduralTransformsNode.creator,
        SomaProceduralTransformsNode.initialize,
        om.MPxNode.kDependNode,
    )
    plugin_fn.registerCommand(
        COMMAND_NAME,
        SomaCreateProceduralRigReferenceCommand.creator,
        SomaCreateProceduralRigReferenceCommand.syntax_creator,
    )


def uninitializePlugin(plugin):
    plugin_fn = om.MFnPlugin(plugin)
    plugin_fn.deregisterCommand(COMMAND_NAME)
    plugin_fn.deregisterNode(NODE_TYPE_ID)
