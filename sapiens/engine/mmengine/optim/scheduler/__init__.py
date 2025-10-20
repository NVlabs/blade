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

# yapf: disable
from .lr_scheduler import (ConstantLR, CosineAnnealingLR, CosineRestartLR,
                           ExponentialLR, LinearLR, MultiStepLR, OneCycleLR,
                           PolyLR, ReduceOnPlateauLR, StepLR)
from .momentum_scheduler import (ConstantMomentum, CosineAnnealingMomentum,
                                 CosineRestartMomentum, ExponentialMomentum,
                                 LinearMomentum, MultiStepMomentum,
                                 PolyMomentum, ReduceOnPlateauMomentum,
                                 StepMomentum)
from .param_scheduler import (ConstantParamScheduler,
                              CosineAnnealingParamScheduler,
                              CosineRestartParamScheduler,
                              ExponentialParamScheduler, LinearParamScheduler,
                              MultiStepParamScheduler, OneCycleParamScheduler,
                              PolyParamScheduler,
                              ReduceOnPlateauParamScheduler,
                              StepParamScheduler, _ParamScheduler)

# yapf: enable
__all__ = [
    'ConstantLR', 'CosineAnnealingLR', 'ExponentialLR', 'LinearLR',
    'MultiStepLR', 'StepLR', 'ConstantMomentum', 'CosineAnnealingMomentum',
    'ExponentialMomentum', 'LinearMomentum', 'MultiStepMomentum',
    'StepMomentum', 'ConstantParamScheduler', 'CosineAnnealingParamScheduler',
    'ExponentialParamScheduler', 'LinearParamScheduler',
    'MultiStepParamScheduler', 'StepParamScheduler', '_ParamScheduler',
    'PolyParamScheduler', 'PolyLR', 'PolyMomentum', 'OneCycleParamScheduler',
    'OneCycleLR', 'CosineRestartParamScheduler', 'CosineRestartLR',
    'CosineRestartMomentum', 'ReduceOnPlateauParamScheduler',
    'ReduceOnPlateauLR', 'ReduceOnPlateauMomentum'
]
