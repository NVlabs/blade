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

from .build_functions import (build_from_cfg, build_model_from_cfg,
                              build_runner_from_cfg, build_scheduler_from_cfg)
from .default_scope import DefaultScope
from .registry import Registry
from .root import (DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS, HOOKS,
                   INFERENCERS, LOG_PROCESSORS, LOOPS, METRICS, MODEL_WRAPPERS,
                   MODELS, OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS,
                   OPTIMIZERS, PARAM_SCHEDULERS, RUNNER_CONSTRUCTORS, RUNNERS,
                   STRATEGIES, TASK_UTILS, TRANSFORMS, VISBACKENDS,
                   VISUALIZERS, WEIGHT_INITIALIZERS)
from .utils import (count_registered_modules, init_default_scope,
                    traverse_registry_tree)

__all__ = [
    'Registry', 'RUNNERS', 'RUNNER_CONSTRUCTORS', 'HOOKS', 'DATASETS',
    'DATA_SAMPLERS', 'TRANSFORMS', 'MODELS', 'WEIGHT_INITIALIZERS',
    'OPTIMIZERS', 'OPTIM_WRAPPER_CONSTRUCTORS', 'TASK_UTILS',
    'PARAM_SCHEDULERS', 'METRICS', 'MODEL_WRAPPERS', 'OPTIM_WRAPPERS', 'LOOPS',
    'VISBACKENDS', 'VISUALIZERS', 'LOG_PROCESSORS', 'EVALUATOR', 'INFERENCERS',
    'DefaultScope', 'traverse_registry_tree', 'count_registered_modules',
    'build_model_from_cfg', 'build_runner_from_cfg', 'build_from_cfg',
    'build_scheduler_from_cfg', 'init_default_scope', 'FUNCTIONS', 'STRATEGIES'
]
