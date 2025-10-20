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

from mmcv.transforms import (CenterCrop, LoadImageFromFile, Normalize,
                             RandomFlip, RandomGrayscale, RandomResize, Resize)

from mmpretrain.registry import TRANSFORMS
from .auto_augment import (AutoAugment, AutoContrast, BaseAugTransform,
                           Brightness, ColorTransform, Contrast, Cutout,
                           Equalize, GaussianBlur, Invert, Posterize,
                           RandAugment, Rotate, Sharpness, Shear, Solarize,
                           SolarizeAdd, Translate)
from .formatting import (Collect, NumpyToPIL, PackInputs, PackMultiTaskInputs,
                         PILToNumpy, Transpose)
from .processing import (Albumentations, BEiTMaskGenerator, CleanCaption,
                         ColorJitter, EfficientNetCenterCrop,
                         EfficientNetRandomCrop, Lighting,
                         MAERandomResizedCrop, RandomCrop, RandomErasing,
                         RandomResizedCrop,
                         RandomResizedCropAndInterpolationWithTwoPic,
                         RandomTranslatePad, ResizeEdge, SimMIMMaskGenerator)
from .utils import get_transform_idx, remove_transform
from .wrappers import ApplyToList, MultiView

for t in (CenterCrop, LoadImageFromFile, Normalize, RandomFlip,
          RandomGrayscale, RandomResize, Resize):
    TRANSFORMS.register_module(module=t)

__all__ = [
    'NumpyToPIL', 'PILToNumpy', 'Transpose', 'Collect', 'RandomCrop',
    'RandomResizedCrop', 'Shear', 'Translate', 'Rotate', 'Invert',
    'ColorTransform', 'Solarize', 'Posterize', 'AutoContrast', 'Equalize',
    'Contrast', 'Brightness', 'Sharpness', 'AutoAugment', 'SolarizeAdd',
    'Cutout', 'RandAugment', 'Lighting', 'ColorJitter', 'RandomErasing',
    'PackInputs', 'Albumentations', 'EfficientNetRandomCrop',
    'EfficientNetCenterCrop', 'ResizeEdge', 'BaseAugTransform',
    'PackMultiTaskInputs', 'GaussianBlur', 'BEiTMaskGenerator',
    'SimMIMMaskGenerator', 'CenterCrop', 'LoadImageFromFile', 'Normalize',
    'RandomFlip', 'RandomGrayscale', 'RandomResize', 'Resize', 'MultiView',
    'ApplyToList', 'CleanCaption', 'RandomTranslatePad',
    'RandomResizedCropAndInterpolationWithTwoPic', 'get_transform_idx',
    'remove_transform', 'MAERandomResizedCrop'
]
