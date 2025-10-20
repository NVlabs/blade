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
from mmcv.utils import Registry

from .base_renderer import BaseRenderer
from .depth_renderer import DepthRenderer
from .mesh_renderer import MeshRenderer
from .normal_renderer import NormalRenderer
from .pointcloud_renderer import PointCloudRenderer
from .segmentation_renderer import SegmentationRenderer
from .silhouette_renderer import SilhouetteRenderer
from .uv_renderer import UVRenderer

RENDERER = Registry('renderer')
RENDERER.register_module(
    name=['base', 'Base', 'base_renderer', 'BaseRenderer'],
    module=BaseRenderer)
RENDERER.register_module(
    name=['Depth', 'depth', 'depth_renderer', 'DepthRenderer'],
    module=DepthRenderer)
RENDERER.register_module(
    name=['Mesh', 'mesh', 'mesh_renderer', 'MeshRenderer'],
    module=MeshRenderer)
RENDERER.register_module(
    name=['Normal', 'normal', 'normal_renderer', 'NormalRenderer'],
    module=NormalRenderer)
RENDERER.register_module(
    name=[
        'PointCloud', 'pointcloud', 'point_cloud', 'pointcloud_renderer',
        'PointCloudRenderer'
    ],
    module=PointCloudRenderer)
RENDERER.register_module(
    name=[
        'segmentation', 'segmentation_renderer', 'Segmentation',
        'SegmentationRenderer'
    ],
    module=SegmentationRenderer)
RENDERER.register_module(
    name=[
        'silhouette', 'silhouette_renderer', 'Silhouette', 'SilhouetteRenderer'
    ],
    module=SilhouetteRenderer)
RENDERER.register_module(
    name=['uv_renderer', 'uv', 'UV', 'UVRenderer'], module=UVRenderer)


def build_renderer(cfg):
    """Build renderers."""
    return RENDERER.build(cfg)
