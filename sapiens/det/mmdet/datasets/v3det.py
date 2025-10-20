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

import os.path
from typing import Optional

import mmengine

from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class V3DetDataset(CocoDataset):
    """Dataset for V3Det."""

    METAINFO = {
        'classes': None,
        'palette': None,
    }

    def __init__(
            self,
            *args,
            metainfo: Optional[dict] = None,
            data_root: str = '',
            label_file='annotations/category_name_13204_v3det_2023_v1.txt',  # noqa
            **kwargs) -> None:
        class_names = tuple(
            mmengine.list_from_file(os.path.join(data_root, label_file)))
        if metainfo is None:
            metainfo = {'classes': class_names}
        super().__init__(
            *args, data_root=data_root, metainfo=metainfo, **kwargs)
