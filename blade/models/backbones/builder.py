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
from mmhuman3d.models.backbones.builder import BACKBONES
from .resnet import ResNet
from .hrnet import (PoseHighResolutionNet, PoseHighResolutionNetExpose,
                    PoseHighResolutionNetGraphormer)
# from .vision_transformer import VisionTransformer

BACKBONES.register_module(name='ResNet', module=ResNet, force=True)
BACKBONES.register_module(name='PoseHighResolutionNet',
                          module=PoseHighResolutionNet,
                          force=True)
BACKBONES.register_module(name='PoseHighResolutionNetExpose',
                          module=PoseHighResolutionNetExpose,
                          force=True)

BACKBONES.register_module(name='PoseHighResolutionNetGraphormer',
                          module=PoseHighResolutionNetGraphormer,
                          force=True)

# BACKBONES.register_module(name='VisionTransformer',
#                           module=VisionTransformer,
#                           force=True)


def build_backbone(cfg):
    """Build head."""
    if cfg is None:
        return None
    return BACKBONES.build(cfg)
