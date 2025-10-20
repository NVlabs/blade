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

import copy
from typing import List, Union

from mmcv.transforms import BaseTransform

PIPELINE_TYPE = List[Union[dict, BaseTransform]]


def get_transform_idx(pipeline: PIPELINE_TYPE, target: str) -> int:
    """Returns the index of the transform in a pipeline.

    Args:
        pipeline (List[dict] | List[BaseTransform]): The transforms list.
        target (str): The target transform class name.

    Returns:
        int: The transform index. Returns -1 if not found.
    """
    for i, transform in enumerate(pipeline):
        if isinstance(transform, dict):
            if isinstance(transform['type'], type):
                if transform['type'].__name__ == target:
                    return i
            else:
                if transform['type'] == target:
                    return i
        else:
            if transform.__class__.__name__ == target:
                return i

    return -1


def remove_transform(pipeline: PIPELINE_TYPE, target: str, inplace=False):
    """Remove the target transform type from the pipeline.

    Args:
        pipeline (List[dict] | List[BaseTransform]): The transforms list.
        target (str): The target transform class name.
        inplace (bool): Whether to modify the pipeline inplace.

    Returns:
        The modified transform.
    """
    idx = get_transform_idx(pipeline, target)
    if not inplace:
        pipeline = copy.deepcopy(pipeline)
    while idx >= 0:
        pipeline.pop(idx)
        idx = get_transform_idx(pipeline, target)

    return pipeline
