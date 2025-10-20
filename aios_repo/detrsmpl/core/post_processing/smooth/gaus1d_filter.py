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
from scipy.ndimage.filters import gaussian_filter1d

from ..builder import POST_PROCESSING


@POST_PROCESSING.register_module(name=['Gaus1dFilter', 'gaus1d'])
class Gaus1dFilter:
    """Applies median filter and then gaussian filter. code from:
    https://github.com/akanazawa/human_dynamics/blob/mas
    ter/src/util/smooth_bbox.py.

    Args:
        x (np.ndarray): input pose
        window_size (int, optional): for median filters (must be odd).
        sigma (float, optional): Sigma for gaussian smoothing.

    Returns:
        np.ndarray: Smoothed poses
    """
    def __init__(self, window_size=11, sigma=4):
        super(Gaus1dFilter, self).__init__()

        self.window_size = window_size
        self.sigma = sigma

    def __call__(self, x=None):
        if self.window_size % 2 == 0:
            window_size = self.window_size - 1
        else:
            window_size = self.window_size
        if window_size > x.shape[0]:
            window_size = x.shape[0]
        if len(x.shape) != 3:
            warnings.warn('x should be a tensor or numpy of [T*M,K,C]')
        assert len(x.shape) == 3
        x_type = x
        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                x = x.cpu().numpy()
            else:
                x = x.numpy()
        # smoothed = np.array(
        #     [signal.medfilt(param, window_size) for param in x.T]).T
        # smooth_poses = np.array(
        #     [gaussian_filter1d(traj, self.sigma) for traj in smoothed.T]).T
        smoothed = np.zeros_like(x)
        for k in range(x.shape[1]):
            for c in range(x.shape[2]):
                smoothed[:, k, c] = signal.medfilt(x[:, k, c], kernel_size=window_size)
                
        smooth_poses = np.zeros_like(smoothed)
        for k in range(smoothed.shape[1]):
            for c in range(smoothed.shape[2]):
                smooth_poses[:, k, c] = gaussian_filter1d(smoothed[:, k, c], sigma=self.sigma)


        if isinstance(x_type, torch.Tensor):
            # we also return tensor by default
            if x_type.is_cuda:
                smooth_poses = torch.from_numpy(smooth_poses).cuda()
            else:
                smooth_poses = torch.from_numpy(smooth_poses)

        return smooth_poses
