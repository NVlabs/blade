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
import pytest
import torch

from mmcv.ops import diff_iou_rotated_2d, diff_iou_rotated_3d


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_diff_iou_rotated_2d():
    np_boxes1 = np.asarray([[[0.5, 0.5, 1., 1., .0], [0.5, 0.5, 1., 1., .0],
                             [0.5, 0.5, 1., 1., .0], [0.5, 0.5, 1., 1., .0],
                             [0.5, 0.5, 1., 1., .0]]],
                           dtype=np.float32)
    np_boxes2 = np.asarray(
        [[[0.5, 0.5, 1., 1., .0], [0.5, 0.5, 1., 1., np.pi / 2],
          [0.5, 0.5, 1., 1., np.pi / 4], [1., 1., 1., 1., .0],
          [1.5, 1.5, 1., 1., .0]]],
        dtype=np.float32)

    boxes1 = torch.from_numpy(np_boxes1).cuda()
    boxes2 = torch.from_numpy(np_boxes2).cuda()

    np_expect_ious = np.asarray([[1., 1., .7071, 1 / 7, .0]])
    ious = diff_iou_rotated_2d(boxes1, boxes2)
    assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_diff_iou_rotated_3d():
    np_boxes1 = np.asarray(
        [[[.5, .5, .5, 1., 1., 1., .0], [.5, .5, .5, 1., 1., 1., .0],
          [.5, .5, .5, 1., 1., 1., .0], [.5, .5, .5, 1., 1., 1., .0],
          [.5, .5, .5, 1., 1., 1., .0]]],
        dtype=np.float32)
    np_boxes2 = np.asarray(
        [[[.5, .5, .5, 1., 1., 1., .0], [.5, .5, .5, 1., 1., 2., np.pi / 2],
          [.5, .5, .5, 1., 1., 1., np.pi / 4], [1., 1., 1., 1., 1., 1., .0],
          [-1.5, -1.5, -1.5, 2.5, 2.5, 2.5, .0]]],
        dtype=np.float32)

    boxes1 = torch.from_numpy(np_boxes1).cuda()
    boxes2 = torch.from_numpy(np_boxes2).cuda()

    np_expect_ious = np.asarray([[1., .5, .7071, 1 / 15, .0]])
    ious = diff_iou_rotated_3d(boxes1, boxes2)
    assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)
