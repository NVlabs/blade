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
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class iSAIDDataset(CocoDataset):
    """Dataset for iSAID instance segmentation.

    iSAID: A Large-scale Dataset for Instance Segmentation
    in Aerial Images.

    For more detail, please refer to "projects/iSAID/README.md"
    """

    METAINFO = dict(
        classes=('background', 'ship', 'store_tank', 'baseball_diamond',
                 'tennis_court', 'basketball_court', 'Ground_Track_Field',
                 'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
                 'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane',
                 'Harbor'),
        palette=[(0, 0, 0), (0, 0, 63), (0, 63, 63), (0, 63, 0), (0, 63, 127),
                 (0, 63, 191), (0, 63, 255), (0, 127, 63), (0, 127, 127),
                 (0, 0, 127), (0, 0, 191), (0, 0, 255), (0, 191, 127),
                 (0, 127, 191), (0, 127, 255), (0, 100, 155)])
