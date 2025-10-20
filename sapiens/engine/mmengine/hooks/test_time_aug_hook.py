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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mmengine.runner import Runner

from mmengine.hooks import Hook
from mmengine.registry import HOOKS, MODELS, RUNNERS


@HOOKS.register_module()
class PrepareTTAHook(Hook):
    """Wraps `runner.model` with subclass of :class:`BaseTTAModel` in
    `before_test`.

    Note:
        This function will only be used with :obj:`MMFullyShardedDataParallel`.

    Args:
        tta_cfg (dict): Config dictionary of the test time augmentation model.
    """

    def __init__(self, tta_cfg: dict):
        self.tta_cfg = tta_cfg

    def before_test(self, runner: 'Runner') -> None:
        """Wraps `runner.model` with the subclass of :class:`BaseTTAModel`.

        Args:
            runner (Runner): The runner of the testing process.
        """
        self.tta_cfg['module'] = runner.model  # type: ignore
        model = MODELS.build(self.tta_cfg)
        runner.model = model  # type: ignore


def build_runner_with_tta(cfg: dict) -> 'Runner':
    """Builds runner with tta (test time augmentation) transformation and
    TTAModel.

    Note:
        This function will only be used with :obj:`MMFullyShardedDataParallel`.

    Args:
        cfg (dict): cfg with ``tta_pipeline`` and ``tta_model``

    Notes:
        This is only an experimental feature. We may refactor the code in the
        future.

    Returns:
        Runner: Runner with tta transformation and TTAModel
    """
    assert hasattr(
        cfg,
        'tta_model'), ('please make sure tta_model is defined in your config.')
    assert hasattr(cfg, 'tta_pipeline'), (
        'please make sure tta_pipeline is defined in your config.')
    cfg['test_dataloader']['dataset']['pipeline'] = cfg['tta_pipeline']

    if 'runner_type' in cfg:
        runner = RUNNERS.build(cfg)
    else:
        from mmengine.runner import Runner
        runner = Runner.from_cfg(cfg)

    runner.register_hook(PrepareTTAHook(tta_cfg=cfg['tta_model']))
    return runner
