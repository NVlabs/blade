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
import torch, os


def keypoint_xyxyzz_to_xyzxyz(keypoints: torch.Tensor):
    """_summary_

    Args:
        keypoints (torch.Tensor): ..., 51
    """
    res = torch.zeros_like(keypoints)
    num_points = keypoints.shape[-1] // 3
    Z = keypoints[..., :2 * num_points]
    V = keypoints[..., 2 * num_points:]
    res[..., 0::3] = Z[..., 0::2]
    res[..., 1::3] = Z[..., 1::2]
    res[..., 2::3] = V[...]
    return res


def keypoint_xyzxyz_to_xyxyzz(keypoints: torch.Tensor):
    """_summary_

    Args:
        keypoints (torch.Tensor): ..., 51
    """
    res = torch.zeros_like(keypoints)
    num_points = keypoints.shape[-1] // 3
    res[..., 0:2 * num_points:2] = keypoints[..., 0::3]
    res[..., 1:2 * num_points:2] = keypoints[..., 1::3]
    res[..., 2 * num_points:] = keypoints[..., 2::3]
    return res
