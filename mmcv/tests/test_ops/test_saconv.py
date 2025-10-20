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
import torch
import torch.nn as nn

from mmcv.ops import SAConv2d


def test_sacconv():

    # test with normal cast
    x = torch.rand(1, 3, 256, 256)
    saconv = SAConv2d(3, 5, kernel_size=3, padding=1)
    sac_out = saconv(x)
    refer_conv = nn.Conv2d(3, 5, kernel_size=3, padding=1)
    refer_out = refer_conv(x)
    assert sac_out.shape == refer_out.shape

    # test with dilation >= 2
    dalited_saconv = SAConv2d(3, 5, kernel_size=3, padding=2, dilation=2)
    dalited_sac_out = dalited_saconv(x)
    refer_conv = nn.Conv2d(3, 5, kernel_size=3, padding=2, dilation=2)
    refer_out = refer_conv(x)
    assert dalited_sac_out.shape == refer_out.shape

    # test with deform
    deform_saconv = SAConv2d(3, 5, kernel_size=3, padding=1, use_deform=True)
    if torch.cuda.is_available():
        x = torch.rand(1, 3, 256, 256).cuda()
        deform_saconv = SAConv2d(
            3, 5, kernel_size=3, padding=1, use_deform=True).cuda()
        deform_sac_out = deform_saconv(x).cuda()
        refer_conv = nn.Conv2d(3, 5, kernel_size=3, padding=1).cuda()
        refer_out = refer_conv(x)
        assert deform_sac_out.shape == refer_out.shape
    else:
        deform_sac_out = deform_saconv(x)
        refer_conv = nn.Conv2d(3, 5, kernel_size=3, padding=1)
        refer_out = refer_conv(x)
        assert deform_sac_out.shape == refer_out.shape

    # test with groups >= 2
    x = torch.rand(1, 4, 256, 256)
    group_saconv = SAConv2d(4, 4, kernel_size=3, padding=1, groups=2)
    group_sac_out = group_saconv(x)
    refer_conv = nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=2)
    refer_out = refer_conv(x)
    assert group_sac_out.shape == refer_out.shape
