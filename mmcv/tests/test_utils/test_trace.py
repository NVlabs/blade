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

from mmcv.utils import digit_version, is_jit_tracing


@pytest.mark.skipif(
    digit_version(torch.__version__) < digit_version('1.6.0'),
    reason='torch.jit.is_tracing is not available before 1.6.0')
def test_is_jit_tracing():

    def foo(x):
        if is_jit_tracing():
            return x
        else:
            return x.tolist()

    x = torch.rand(3)
    # test without trace
    assert isinstance(foo(x), list)

    # test with trace
    traced_foo = torch.jit.trace(foo, (torch.rand(1), ))
    assert isinstance(traced_foo(x), torch.Tensor)
