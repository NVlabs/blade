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
from mmcv.utils import Registry
from pytorch3d.renderer import (
    HardFlatShader,
    HardGouraudShader,
    HardPhongShader,
    SoftGouraudShader,
    SoftPhongShader,
)

from .shader import (
    DepthShader,
    NoLightShader,
    NormalShader,
    SegmentationShader,
    SilhouetteShader,
)

SHADER = Registry('shader')
SHADER.register_module(
    name=[
        'flat', 'hard_flat_shader', 'hard_flat', 'HardFlat', 'HardFlatShader'
    ],
    module=HardFlatShader)
SHADER.register_module(
    name=['hard_phong', 'HardPhong', 'HardPhongShader'],
    module=HardPhongShader)
SHADER.register_module(
    name=['hard_gouraud', 'HardGouraud', 'HardGouraudShader'],
    module=HardGouraudShader)
SHADER.register_module(
    name=['soft_gouraud', 'SoftGouraud', 'SoftGouraudShader'],
    module=SoftGouraudShader)
SHADER.register_module(
    name=['soft_phong', 'SoftPhong', 'SoftPhongShader'],
    module=SoftPhongShader)
SHADER.register_module(
    name=['silhouette', 'Silhouette', 'SilhouetteShader'],
    module=SilhouetteShader)
SHADER.register_module(
    name=['nolight', 'nolight_shader', 'NoLight', 'NoLightShader'],
    module=NoLightShader)
SHADER.register_module(
    name=['normal', 'normal_shader', 'Normal', 'NormalShader'],
    module=NormalShader)
SHADER.register_module(
    name=['depth', 'depth_shader', 'Depth', 'DepthShader'], module=DepthShader)
SHADER.register_module(
    name=[
        'segmentation', 'segmentation_shader', 'Segmentation',
        'SegmentationShader'
    ],
    module=SegmentationShader)


def build_shader(cfg):
    """Build shader."""
    return SHADER.build(cfg)
