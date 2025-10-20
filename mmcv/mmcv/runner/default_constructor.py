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
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from .builder import RUNNER_BUILDERS, RUNNERS


@RUNNER_BUILDERS.register_module()
class DefaultRunnerConstructor:
    """Default constructor for runners.

    Custom existing `Runner` like `EpocBasedRunner` though `RunnerConstructor`.
    For example, We can inject some new properties and functions for `Runner`.

    Example:
        >>> from mmcv.runner import RUNNER_BUILDERS, build_runner
        >>> # Define a new RunnerReconstructor
        >>> @RUNNER_BUILDERS.register_module()
        >>> class MyRunnerConstructor:
        ...     def __init__(self, runner_cfg, default_args=None):
        ...         if not isinstance(runner_cfg, dict):
        ...             raise TypeError('runner_cfg should be a dict',
        ...                             f'but got {type(runner_cfg)}')
        ...         self.runner_cfg = runner_cfg
        ...         self.default_args = default_args
        ...
        ...     def __call__(self):
        ...         runner = RUNNERS.build(self.runner_cfg,
        ...                                default_args=self.default_args)
        ...         # Add new properties for existing runner
        ...         runner.my_name = 'my_runner'
        ...         runner.my_function = lambda self: print(self.my_name)
        ...         ...
        >>> # build your runner
        >>> runner_cfg = dict(type='EpochBasedRunner', max_epochs=40,
        ...                   constructor='MyRunnerConstructor')
        >>> runner = build_runner(runner_cfg)
    """

    def __init__(self, runner_cfg: dict, default_args: Optional[dict] = None):
        if not isinstance(runner_cfg, dict):
            raise TypeError('runner_cfg should be a dict',
                            f'but got {type(runner_cfg)}')
        self.runner_cfg = runner_cfg
        self.default_args = default_args

    def __call__(self):
        return RUNNERS.build(self.runner_cfg, default_args=self.default_args)
