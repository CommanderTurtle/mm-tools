# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

MMTOOLS_SOURCE_REVISION = "CommanderTurtle/archive--smplx@1265df7ba545e8b00f72e7c557c766e15c71632f"

from .body_models import (
    create,
    SMPL,
    SMPLH,
    SMPLX,
    MANO,
    FLAME,
    build_layer,
    SMPLLayer,
    SMPLHLayer,
    SMPLXLayer,
    MANOLayer,
    FLAMELayer,
)
