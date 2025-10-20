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
import torch.nn as nn


class Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.view(-1)
        target = target.view(-1)
        return torch.mean(input - target)


class TestCrissCrossAttention:

    def test_cc_attention(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        from mmcv.ops import CrissCrossAttention
        loss_func = Loss()

        input = np.fromfile(
            'tests/data/for_ccattention/ccattention_input.bin',
            dtype=np.float32)
        output = np.fromfile(
            'tests/data/for_ccattention/ccattention_output.bin',
            dtype=np.float32)
        input = input.reshape((1, 32, 45, 45))
        output = output.reshape((1, 32, 45, 45))
        label = torch.ones((1, 32, 45, 45))

        input = torch.FloatTensor(input)
        output = torch.FloatTensor(output)

        input.requires_grad = True

        shape = input.shape
        channel = shape[1]

        cca = CrissCrossAttention(channel)
        cca.to(device)
        input = input.to(device)
        label = label.to(device)
        cca.train()
        test_output = cca(input)
        test_loss = loss_func(test_output, label)
        test_loss.backward()
        test_output = test_output.detach().cpu().numpy()
        output = output.numpy()

        assert np.allclose(test_output, output)
        assert test_output.shape == shape
