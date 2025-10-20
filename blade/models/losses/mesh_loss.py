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
import torch.nn as nn
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)


class ChamferLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, **kwargs):
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        reduction = 'none' if reduction is None else reduction
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss = self.loss_weight * chamfer_distance(
            pred,
            target,
            batch_reduction=self.reduction,
            point_reduction=self.reduction)
        return loss


class MeshEdgeLoss(nn.Module):

    def __init__(self, loss_weight=1.0, target_length=0., **kwargs):
        super().__init__()

        self.loss_weight = loss_weight
        self.target_length = target_length

    def forward(self, mesh):

        loss = self.loss_weight * mesh_edge_loss(
            mesh, target_length=self.target_length)
        return loss


class LaplacianLoss(nn.Module):

    def __init__(self, loss_weight=1.0, method='uniform', **kwargs):
        super().__init__()
        self.method = method
        self.loss_weight = loss_weight

    def forward(self, mesh):
        loss = self.loss_weight * mesh_laplacian_smoothing(mesh,
                                                           method=self.method)
        return loss


class NormalConsistencyLoss(nn.Module):

    def __init__(self, loss_weight=1.0, **kwargs):
        super().__init__()

        self.loss_weight = loss_weight

    def forward(self, mesh):
        loss = self.loss_weight * mesh_normal_consistency(mesh)
        return loss
