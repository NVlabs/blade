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
import torch.nn.functional as F


class TestBilinearGridSample:

    def _test_bilinear_grid_sample(self,
                                   dtype=torch.float,
                                   align_corners=False,
                                   multiplier=1,
                                   precision=1e-3):
        from mmcv.ops.point_sample import bilinear_grid_sample

        input = torch.rand(1, 1, 20, 20, dtype=dtype)
        grid = torch.Tensor([[[1, 0, 0], [0, 1, 0]]])
        grid = F.affine_grid(
            grid, (1, 1, 15, 15), align_corners=align_corners).type_as(input)
        grid *= multiplier

        out = bilinear_grid_sample(input, grid, align_corners=align_corners)
        ref_out = F.grid_sample(input, grid, align_corners=align_corners)

        assert np.allclose(out.data.detach().cpu().numpy(),
                           ref_out.data.detach().cpu().numpy(), precision)

    def test_bilinear_grid_sample(self):
        self._test_bilinear_grid_sample(torch.double, False)
        self._test_bilinear_grid_sample(torch.double, True)
        self._test_bilinear_grid_sample(torch.float, False)
        self._test_bilinear_grid_sample(torch.float, True)
        self._test_bilinear_grid_sample(torch.float, False)
        self._test_bilinear_grid_sample(torch.float, True, 5)
        self._test_bilinear_grid_sample(torch.float, False, 10)
        self._test_bilinear_grid_sample(torch.float, True, -6)
        self._test_bilinear_grid_sample(torch.float, False, -10)
        self._test_bilinear_grid_sample(torch.double, True, 5)
        self._test_bilinear_grid_sample(torch.double, False, 10)
        self._test_bilinear_grid_sample(torch.double, True, -6)
        self._test_bilinear_grid_sample(torch.double, False, -10)
