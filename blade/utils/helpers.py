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
import torch as th, os, numpy as np
import matplotlib.pyplot as plt


import warnings

import torch.nn.functional as F


def resize(input, size=None, scale_factor=None, mode="nearest", align_corners=None, warning=False):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def get_global_rank():
    if not th.distributed.is_available():
        print("torch distributed is not available")
        return 0
    if not th.distributed.is_initialized():
        print("torch distributed is not initialized")
        return 0
    return th.distributed.get_rank()


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


# Function to perform perspective projection
def perspective_projection(vertices, focal_length):
    # Assume the camera is at the origin looking along the Z-axis
    projected = vertices.clone()
    projected[..., 0] = (focal_length * vertices[..., 0]) / vertices[..., 2] #+ image_width / 2
    projected[..., 1] = (focal_length * vertices[..., 1]) / vertices[..., 2] #+ image_height / 2
    return projected[..., :2]


# Function to perform orthographic projection
def orthographic_projection(vertices):
    projected = vertices.clone()
    projected[..., 0] = vertices[..., 0] # + image_width / 2
    projected[..., 1] = vertices[..., 1] # + image_height / 2
    return projected[..., :2]


# Function to calculate the optimal scale and translation
def optimize_scale_and_translation(ortho_proj, perspective_proj):
    # def objective(params):
    #     scale = params
    #     transformed = ortho_proj * scale
    #     # transformed[:, 0] += tx + imw/2
    #     # transformed[:, 1] += ty + imh/2
    #     return np.mean((transformed - perspective_proj) ** 2)
    #
    # initial_guess = [1.0]
    # result = minimize(objective, initial_guess, method='BFGS')
    ortho_scale = th.mean(th.norm(ortho_proj, dim=1))
    persp_scale = th.mean(th.norm(perspective_proj, dim=1))

    return persp_scale / ortho_scale


def calculate_distortion_error(perspective_proj, aligned_ortho_proj):
    # Calculate the difference
    difference = th.norm(aligned_ortho_proj - perspective_proj, dim=-1).mean(dim=-1)

    return difference


def vis_projection(perspective_proj, aligned_ortho_proj, z):
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'Tz={z}m')
    ax = plt.subplot(1, 2, 1)
    plt.scatter(perspective_proj[:, 0], perspective_proj[:, 1], c='r', s=2, label='Perspective')
    plt.title('Perspective Projection Aligned to Ortho')
    # plt.xlim(0, image_width)
    # plt.ylim(0, image_height)
    ax.set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.legend()

    ax = plt.subplot(1, 2, 2)
    plt.scatter(aligned_ortho_proj[:, 0], aligned_ortho_proj[:, 1], c='b', s=1, label='Aligned Orthographic')
    plt.title('Orthographic Projection')
    # plt.xlim(0, image_width)
    # plt.ylim(0, image_height)
    ax.set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.legend()

    plt.tight_layout()
    plt.show()

