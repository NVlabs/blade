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
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .norm import build_norm_layer

try:
    from mmdet.models.backbones import ResNet
    from mmdet.models.roi_heads.shared_heads.res_layer import ResLayer
    from mmdet.registry import MODELS

    @MODELS.register_module()
    class ResLayerExtraNorm(ResLayer):
        """Add extra norm to original ``ResLayer``."""

        def __init__(self, *args, **kwargs):
            super(ResLayerExtraNorm, self).__init__(*args, **kwargs)

            block = ResNet.arch_settings[kwargs['depth']][0]
            self.add_module(
                'norm',
                build_norm_layer(self.norm_cfg,
                                 64 * 2**self.stage * block.expansion))

        def forward(self, x):
            """Forward function."""
            res_layer = getattr(self, f'layer{self.stage + 1}')
            norm = getattr(self, 'norm')
            x = res_layer(x)
            out = norm(x)
            return out

except ImportError:
    ResLayerExtraNorm = None
