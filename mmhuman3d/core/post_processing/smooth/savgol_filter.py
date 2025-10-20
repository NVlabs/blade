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
import warnings

import numpy as np
import scipy.signal as signal
import torch

from ..builder import POST_PROCESSING


@POST_PROCESSING.register_module(name=['SGFilter', 'savgol'])
class SGFilter:
    """savgol_filter lib is from:
    https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.signal.savgol_filter.html.

    Args:
        window_size (float):
                    The length of the filter window
                    (i.e., the number of coefficients).
                    window_length must be a positive odd integer.
        polyorder (int):
                    The order of the polynomial used to fit the samples.
                    polyorder must be less than window_length.

    Returns:
        smoothed poses (np.ndarray, torch.tensor)
    """

    def __init__(self, window_size=11, polyorder=2):
        super(SGFilter, self).__init__()

        # 1-D Savitzky-Golay filter
        self.window_size = window_size
        self.polyorder = polyorder

    def __call__(self, x=None):
        # x.shape: [t,k,c]
        if self.window_size % 2 == 0:
            window_size = self.window_size - 1
        else:
            window_size = self.window_size
        if window_size > x.shape[0]:
            window_size = x.shape[0]
        if window_size <= self.polyorder:
            polyorder = window_size - 1
        else:
            polyorder = self.polyorder
        assert polyorder > 0
        assert window_size > polyorder
        if len(x.shape) != 3:
            warnings.warn('x should be a tensor or numpy of [T*M,K,C]')
        assert len(x.shape) == 3
        x_type = x
        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                x = x.cpu().numpy()
            else:
                x = x.numpy()
        smooth_poses = np.zeros_like(x)
        # smooth at different axis
        C = x.shape[-1]
        for i in range(C):
            smooth_poses[..., i] = signal.savgol_filter(
                x[..., i], window_size, polyorder, axis=0)

        if isinstance(x_type, torch.Tensor):
            # we also return tensor by default
            if x_type.is_cuda:
                smooth_poses = torch.from_numpy(smooth_poses).cuda()
            else:
                smooth_poses = torch.from_numpy(smooth_poses)
        return smooth_poses
