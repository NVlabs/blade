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
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch


def test_contour_expand():
    from mmcv.ops import contour_expand

    np_internal_kernel_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 1, 1, 0, 0, 0, 0, 2, 0],
                                         [0, 0, 1, 1, 0, 0, 0, 0, 2, 0],
                                         [0, 0, 1, 1, 0, 0, 0, 0, 2, 0],
                                         [0, 0, 1, 1, 0, 0, 0, 0, 2, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0]]).astype(np.int32)
    np_kernel_mask1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0]]).astype(np.uint8)
    np_kernel_mask2 = (np_internal_kernel_label > 0).astype(np.uint8)

    np_kernel_mask = np.stack([np_kernel_mask1, np_kernel_mask2])
    min_area = 1
    kernel_region_num = 3
    result = contour_expand(np_kernel_mask, np_internal_kernel_label, min_area,
                            kernel_region_num)
    gt = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 2, 2, 2, 0],
          [0, 0, 1, 1, 1, 1, 2, 2, 2, 0], [0, 0, 1, 1, 1, 1, 2, 2, 2, 0],
          [0, 0, 1, 1, 1, 1, 2, 2, 2, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    assert np.allclose(result, gt)

    np_kernel_mask_t = torch.from_numpy(np_kernel_mask)
    np_internal_kernel_label_t = torch.from_numpy(np_internal_kernel_label)
    result = contour_expand(np_kernel_mask_t, np_internal_kernel_label_t,
                            min_area, kernel_region_num)
    assert np.allclose(result, gt)
