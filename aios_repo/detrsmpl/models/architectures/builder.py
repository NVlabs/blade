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

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

from .DetrSMPL import MultiBodyEstimator
from .expressive_mesh_estimator import SMPLXImageBodyModelEstimator
from .hybrik import HybrIK_trainer
from .mesh_estimator import ImageBodyModelEstimator, VideoBodyModelEstimator
from .DetrSMPLloss import DETRLoss


def build_from_cfg(cfg, registry, default_args=None):
    if cfg is None:
        return None
    return MMCV_MODELS.build_func(cfg, registry, default_args)


ARCHITECTURES = Registry('architectures',
                         parent=MMCV_MODELS,
                         build_func=build_from_cfg)

ARCHITECTURES.register_module(name='HybrIK_trainer', module=HybrIK_trainer)
ARCHITECTURES.register_module(name='ImageBodyModelEstimator',
                              module=ImageBodyModelEstimator)
ARCHITECTURES.register_module(name='VideoBodyModelEstimator',
                              module=VideoBodyModelEstimator)
ARCHITECTURES.register_module(name='SMPLXImageBodyModelEstimator',
                              module=SMPLXImageBodyModelEstimator)
ARCHITECTURES.register_module(name='MultiBodyEstimator',
                              module=MultiBodyEstimator)
ARCHITECTURES.register_module(name='DETRLoss', module=DETRLoss)


def build_architecture(cfg):
    """Build framework."""
    return ARCHITECTURES.build(cfg)
