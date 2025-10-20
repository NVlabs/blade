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

from .activation import build_activation_layer
from .context_block import ContextBlock
from .conv import build_conv_layer
from .conv2d_adaptive_padding import Conv2dAdaptivePadding
from .conv_module import ConvModule
from .conv_ws import ConvAWS2d, ConvWS2d, conv_ws_2d
from .depthwise_separable_conv_module import DepthwiseSeparableConvModule
from .drop import Dropout, DropPath
from .generalized_attention import GeneralizedAttention
from .hsigmoid import HSigmoid
from .hswish import HSwish
from .non_local import NonLocal1d, NonLocal2d, NonLocal3d
from .norm import build_norm_layer, is_norm
from .padding import build_padding_layer
from .plugin import build_plugin_layer
from .scale import LayerScale, Scale
from .swish import Swish
from .upsample import build_upsample_layer
from .wrappers import (Conv2d, Conv3d, ConvTranspose2d, ConvTranspose3d,
                       Linear, MaxPool2d, MaxPool3d)

__all__ = [
    'ConvModule', 'build_activation_layer', 'build_conv_layer',
    'build_norm_layer', 'build_padding_layer', 'build_upsample_layer',
    'build_plugin_layer', 'is_norm', 'HSigmoid', 'HSwish', 'NonLocal1d',
    'NonLocal2d', 'NonLocal3d', 'ContextBlock', 'GeneralizedAttention',
    'Scale', 'ConvAWS2d', 'ConvWS2d', 'conv_ws_2d',
    'DepthwiseSeparableConvModule', 'Swish', 'Linear', 'Conv2dAdaptivePadding',
    'Conv2d', 'ConvTranspose2d', 'MaxPool2d', 'ConvTranspose3d', 'MaxPool3d',
    'Conv3d', 'Dropout', 'DropPath', 'LayerScale'
]
