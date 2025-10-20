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


from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import master_only
import wandb

@HOOKS.register_module()
class WandbEpochLoggerHook(Hook):
    def __init__(self, interval=1, metrics=None, init_kwargs=None):
        self.interval = interval
        self.metrics = metrics
        self.init_kwargs = init_kwargs
        self.wandb = wandb

    @master_only
    def before_run(self, runner):
        super(WandbEpochLoggerHook, self).before_run(runner)
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.wandb.init(**self.init_kwargs)
        else:
            self.wandb.init()

    @master_only
    def after_val_epoch(self, runner):
        if runner.epoch % self.interval == 0:
            # Log evaluation metrics
            metrics = runner.log_buffer.output
            out = {'epoch': runner.epoch}
            for k in self.metrics:
                out[k] =  metrics.get(k, None)
            self.wandb.log()

    @master_only
    def after_run(self, runner):
        self.wandb.join()
        self.wandb.finish()