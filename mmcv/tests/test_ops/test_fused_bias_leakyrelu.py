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
import pytest
import torch

_USING_PARROTS = True
try:
    from parrots.autograd import gradcheck
except ImportError:
    from torch.autograd import gradcheck, gradgradcheck
    _USING_PARROTS = False


class TestFusedBiasLeakyReLU:

    @classmethod
    def setup_class(cls):
        if not torch.cuda.is_available():
            return
        cls.input_tensor = torch.randn((2, 2, 2, 2), requires_grad=True).cuda()
        cls.bias = torch.zeros(2, requires_grad=True).cuda()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_gradient(self):

        from mmcv.ops import FusedBiasLeakyReLU
        if _USING_PARROTS:
            gradcheck(
                FusedBiasLeakyReLU(2).cuda(),
                self.input_tensor,
                delta=1e-4,
                pt_atol=1e-3)
        else:
            gradcheck(
                FusedBiasLeakyReLU(2).cuda(),
                self.input_tensor,
                eps=1e-4,
                atol=1e-3)

    @pytest.mark.skipif(
        not torch.cuda.is_available() or _USING_PARROTS,
        reason='requires cuda')
    def test_gradgradient(self):

        from mmcv.ops import FusedBiasLeakyReLU
        gradgradcheck(
            FusedBiasLeakyReLU(2).cuda(),
            self.input_tensor,
            eps=1e-4,
            atol=1e-3)
