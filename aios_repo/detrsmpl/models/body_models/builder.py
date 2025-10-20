# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling BLADE or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.utils import Registry

from .flame import FLAME, FLAMELayer
from .mano import MANO, MANOLayer
from .smpl import SMPL, GenderedSMPL, HybrIKSMPL
from .smplx import SMPLX, SMPLXLayer
from .star import STAR

BODY_MODELS = Registry('body_models')

BODY_MODELS.register_module(name=['SMPL', 'smpl'], module=SMPL)
BODY_MODELS.register_module(name='GenderedSMPL', module=GenderedSMPL)
BODY_MODELS.register_module(name=['STAR', 'star'], module=STAR)
BODY_MODELS.register_module(
    name=['HybrIKSMPL', 'HybrIKsmpl', 'hybriksmpl', 'hybrik', 'hybrIK'],
    module=HybrIKSMPL)
BODY_MODELS.register_module(name=['SMPLX', 'smplx'], module=SMPLX)
BODY_MODELS.register_module(name=['flame', 'FLAME'], module=FLAME)
BODY_MODELS.register_module(name=['MANO', 'mano'], module=MANO)
BODY_MODELS.register_module(name=['SMPLXLayer', 'smplxlayer'],
                            module=SMPLXLayer)
BODY_MODELS.register_module(name=['MANOLayer', 'manolayer'], module=MANOLayer)
BODY_MODELS.register_module(name=['FLAMELayer', 'flamelayer'],
                            module=FLAMELayer)


def build_body_model(cfg):
    """Build body_models."""
    if cfg is None:
        return None
    return BODY_MODELS.build(cfg)
