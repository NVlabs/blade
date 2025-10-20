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

from .asymmetric_loss import AsymmetricLoss, asymmetric_loss
from .cae_loss import CAELoss
from .cosine_similarity_loss import CosineSimilarityLoss
from .cross_correlation_loss import CrossCorrelationLoss
from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy)
from .focal_loss import FocalLoss, sigmoid_focal_loss
from .label_smooth_loss import LabelSmoothLoss
from .reconstruction_loss import PixelReconstructionLoss
from .seesaw_loss import SeesawLoss
from .swav_loss import SwAVLoss
from .utils import (convert_to_one_hot, reduce_loss, weight_reduce_loss,
                    weighted_loss)

__all__ = [
    'asymmetric_loss',
    'AsymmetricLoss',
    'cross_entropy',
    'binary_cross_entropy',
    'CrossEntropyLoss',
    'reduce_loss',
    'weight_reduce_loss',
    'LabelSmoothLoss',
    'weighted_loss',
    'FocalLoss',
    'sigmoid_focal_loss',
    'convert_to_one_hot',
    'SeesawLoss',
    'CAELoss',
    'CosineSimilarityLoss',
    'CrossCorrelationLoss',
    'PixelReconstructionLoss',
    'SwAVLoss',
]
