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

from .base import BaseTransform
from .builder import TRANSFORMS
from .loading import LoadAnnotations, LoadImageFromFile
from .processing import (CenterCrop, MultiScaleFlipAug, Normalize, Pad,
                         RandomChoiceResize, RandomFlip, RandomGrayscale,
                         RandomResize, Resize, TestTimeAug)
from .wrappers import (Compose, KeyMapper, RandomApply, RandomChoice,
                       TransformBroadcaster)

try:
    import torch  # noqa: F401
except ImportError:
    __all__ = [
        'BaseTransform', 'TRANSFORMS', 'TransformBroadcaster', 'Compose',
        'RandomChoice', 'KeyMapper', 'LoadImageFromFile', 'LoadAnnotations',
        'Normalize', 'Resize', 'Pad', 'RandomFlip', 'RandomChoiceResize',
        'CenterCrop', 'RandomGrayscale', 'MultiScaleFlipAug', 'RandomResize',
        'RandomApply', 'TestTimeAug'
    ]
else:
    from .formatting import ImageToTensor, ToTensor, to_tensor

    __all__ = [
        'BaseTransform', 'TRANSFORMS', 'TransformBroadcaster', 'Compose',
        'RandomChoice', 'KeyMapper', 'LoadImageFromFile', 'LoadAnnotations',
        'Normalize', 'Resize', 'Pad', 'ToTensor', 'to_tensor', 'ImageToTensor',
        'RandomFlip', 'RandomChoiceResize', 'CenterCrop', 'RandomGrayscale',
        'MultiScaleFlipAug', 'RandomResize', 'RandomApply', 'TestTimeAug'
    ]
