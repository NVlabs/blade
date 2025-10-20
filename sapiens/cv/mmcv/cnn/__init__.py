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

from .alexnet import AlexNet
# yapf: disable
from .bricks import (ContextBlock, Conv2d, Conv3d, ConvAWS2d, ConvModule,
                     ConvTranspose2d, ConvTranspose3d, ConvWS2d,
                     DepthwiseSeparableConvModule, GeneralizedAttention,
                     HSigmoid, HSwish, Linear, MaxPool2d, MaxPool3d,
                     NonLocal1d, NonLocal2d, NonLocal3d, Scale, Swish,
                     build_activation_layer, build_conv_layer,
                     build_norm_layer, build_padding_layer, build_plugin_layer,
                     build_upsample_layer, conv_ws_2d, is_norm)
# yapf: enable
from .resnet import ResNet, make_res_layer
from .rfsearch import Conv2dRFSearchOp, RFSearchHook
from .utils import fuse_conv_bn, get_model_complexity_info
from .vgg import VGG, make_vgg_layer

__all__ = [
    'AlexNet', 'VGG', 'make_vgg_layer', 'ResNet', 'make_res_layer',
    'ConvModule', 'build_activation_layer', 'build_conv_layer',
    'build_norm_layer', 'build_padding_layer', 'build_upsample_layer',
    'build_plugin_layer', 'is_norm', 'NonLocal1d', 'NonLocal2d', 'NonLocal3d',
    'ContextBlock', 'HSigmoid', 'Swish', 'HSwish', 'GeneralizedAttention',
    'Scale', 'conv_ws_2d', 'ConvAWS2d', 'ConvWS2d',
    'DepthwiseSeparableConvModule', 'Linear', 'Conv2d', 'ConvTranspose2d',
    'MaxPool2d', 'ConvTranspose3d', 'MaxPool3d', 'Conv3d', 'fuse_conv_bn',
    'get_model_complexity_info', 'Conv2dRFSearchOp', 'RFSearchHook'
]
