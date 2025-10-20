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
from torch.autograd import gradcheck


class TestCarafe:

    def test_carafe_naive_gradcheck(self):
        if not torch.cuda.is_available():
            return
        from mmcv.ops import CARAFENaive
        feat = torch.randn(
            2, 64, 3, 3, requires_grad=True, device='cuda').double()
        mask = torch.randn(
            2, 100, 6, 6, requires_grad=True,
            device='cuda').sigmoid().double()
        gradcheck(CARAFENaive(5, 4, 2), (feat, mask), atol=1e-4, eps=1e-4)

    def test_carafe_gradcheck(self):
        if not torch.cuda.is_available():
            return
        from mmcv.ops import CARAFE
        feat = torch.randn(
            2, 64, 3, 3, requires_grad=True, device='cuda').double()
        mask = torch.randn(
            2, 100, 6, 6, requires_grad=True,
            device='cuda').sigmoid().double()
        gradcheck(CARAFE(5, 4, 2), (feat, mask), atol=1e-4, eps=1e-4)
