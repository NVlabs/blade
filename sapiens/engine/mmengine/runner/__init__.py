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

from ._flexible_runner import FlexibleRunner
from .activation_checkpointing import turn_on_activation_checkpointing
from .amp import autocast
from .base_loop import BaseLoop
from .checkpoint import (CheckpointLoader, find_latest_checkpoint,
                         get_deprecated_model_names, get_external_models,
                         get_mmcls_models, get_state_dict,
                         get_torchvision_models, load_checkpoint,
                         load_state_dict, save_checkpoint, weights_to_cpu)
from .log_processor import LogProcessor
from .loops import EpochBasedTrainLoop, IterBasedTrainLoop, TestLoop, ValLoop
from .priority import Priority, get_priority
from .runner import Runner
from .utils import set_random_seed

__all__ = [
    'BaseLoop', 'load_state_dict', 'get_torchvision_models',
    'get_external_models', 'get_mmcls_models', 'get_deprecated_model_names',
    'CheckpointLoader', 'load_checkpoint', 'weights_to_cpu', 'get_state_dict',
    'save_checkpoint', 'EpochBasedTrainLoop', 'IterBasedTrainLoop', 'ValLoop',
    'TestLoop', 'Runner', 'get_priority', 'Priority', 'find_latest_checkpoint',
    'autocast', 'LogProcessor', 'set_random_seed', 'FlexibleRunner',
    'turn_on_activation_checkpointing'
]
