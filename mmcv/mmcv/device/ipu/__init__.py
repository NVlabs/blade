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
from mmcv.utils import IS_IPU_AVAILABLE

if IS_IPU_AVAILABLE:
    from .dataloader import IPUDataLoader
    from .hook_wrapper import IPUFp16OptimizerHook
    from .model_wrapper import ipu_model_wrapper
    from .runner import IPUBaseRunner, IPUEpochBasedRunner, IPUIterBasedRunner
    from .utils import cfg2options
    __all__ = [
        'cfg2options', 'ipu_model_wrapper', 'IPUFp16OptimizerHook',
        'IPUDataLoader', 'IPUBaseRunner', 'IPUEpochBasedRunner',
        'IPUIterBasedRunner'
    ]
