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


from typing import Optional

import torch
from mmengine.model import BaseModule
from torch import nn

from mmpretrain.registry import MODELS


@MODELS.register_module()
class CosineSimilarityLoss(BaseModule):
    """Cosine similarity loss function.

    Compute the similarity between two features and optimize that similarity as
    loss.

    Args:
        shift_factor (float): The shift factor of cosine similarity.
            Default: 0.0.
        scale_factor (float): The scale factor of cosine similarity.
            Default: 1.0.
    """

    def __init__(self,
                 shift_factor: float = 0.0,
                 scale_factor: float = 1.0) -> None:
        super().__init__()
        self.shift_factor = shift_factor
        self.scale_factor = scale_factor

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward function of cosine similarity loss.

        Args:
            pred (torch.Tensor): The predicted features.
            target (torch.Tensor): The target features.

        Returns:
            torch.Tensor: The cosine similarity loss.
        """
        pred_norm = nn.functional.normalize(pred, dim=-1)
        target_norm = nn.functional.normalize(target, dim=-1)
        loss = self.shift_factor - self.scale_factor * (
            pred_norm * target_norm).sum(dim=-1)

        if mask is None:
            loss = loss.mean()
        else:
            loss = (loss * mask).sum() / mask.sum()
        return loss
