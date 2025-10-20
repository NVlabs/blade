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

from typing import List, Optional

import mmengine

from mmpretrain.registry import METRICS
from mmpretrain.utils import require
from .caption import COCOCaption, save_result

try:
    from pycocoevalcap.eval import COCOEvalCap
    from pycocotools.coco import COCO
except ImportError:
    COCOEvalCap = None
    COCO = None


@METRICS.register_module()
class NocapsSave(COCOCaption):
    """Nocaps evaluation wrapper.

    Save the generated captions and transform into coco format.
    The dumped file can be submitted to the official evluation system.

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Should be modified according to the
            `retrieval_type` for unambiguous results. Defaults to TR.
    """

    @require('pycocoevalcap')
    def __init__(self,
                 save_dir: str = './',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super(COCOCaption, self).__init__(
            collect_device=collect_device, prefix=prefix)
        self.save_dir = save_dir

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.
        """
        mmengine.mkdir_or_exist(self.save_dir)
        save_result(
            result=results,
            result_dir=self.save_dir,
            filename='nocap_pred',
            remove_duplicate='image_id',
        )

        return dict()
