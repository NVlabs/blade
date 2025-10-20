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

import json


def load_json_log(json_log):
    """load and convert json_logs to log_dicts.

    Args:
        json_log (str): The path of the json log file.

    Returns:
        dict: The result dict contains two items, "train" and "val", for
        the training log and validate log.

    Example:
        An example output:

        .. code-block:: python

            {
                'train': [
                    {"lr": 0.1, "time": 0.02, "epoch": 1, "step": 100},
                    {"lr": 0.1, "time": 0.02, "epoch": 1, "step": 200},
                    {"lr": 0.1, "time": 0.02, "epoch": 1, "step": 300},
                    ...
                ]
                'val': [
                    {"accuracy/top1": 32.1, "step": 1},
                    {"accuracy/top1": 50.2, "step": 2},
                    {"accuracy/top1": 60.3, "step": 2},
                    ...
                ]
            }
    """
    log_dict = dict(train=[], val=[])
    with open(json_log, 'r') as log_file:
        for line in log_file:
            log = json.loads(line.strip())
            # A hack trick to determine whether the line is training log.
            mode = 'train' if 'lr' in log else 'val'
            log_dict[mode].append(log)

    return log_dict
