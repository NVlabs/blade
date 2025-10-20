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

from .optimizer import (OPTIM_WRAPPER_CONSTRUCTORS, OPTIMIZERS,
                        AmpOptimWrapper, ApexOptimWrapper, BaseOptimWrapper,
                        DefaultOptimWrapperConstructor, OptimWrapper,
                        OptimWrapperDict, ZeroRedundancyOptimizer,
                        build_optim_wrapper)
# yapf: disable
from .scheduler import (ConstantLR, ConstantMomentum, ConstantParamScheduler,
                        CosineAnnealingLR, CosineAnnealingMomentum,
                        CosineAnnealingParamScheduler, ExponentialLR,
                        ExponentialMomentum, ExponentialParamScheduler,
                        LinearLR, LinearMomentum, LinearParamScheduler,
                        MultiStepLR, MultiStepMomentum,
                        MultiStepParamScheduler, OneCycleLR,
                        OneCycleParamScheduler, PolyLR, PolyMomentum,
                        PolyParamScheduler, ReduceOnPlateauLR,
                        ReduceOnPlateauMomentum, ReduceOnPlateauParamScheduler,
                        StepLR, StepMomentum, StepParamScheduler,
                        _ParamScheduler)

# yapf: enable
__all__ = [
    'OPTIM_WRAPPER_CONSTRUCTORS', 'OPTIMIZERS', 'build_optim_wrapper',
    'DefaultOptimWrapperConstructor', 'ConstantLR', 'CosineAnnealingLR',
    'ExponentialLR', 'LinearLR', 'MultiStepLR', 'StepLR', 'ConstantMomentum',
    'CosineAnnealingMomentum', 'ExponentialMomentum', 'LinearMomentum',
    'MultiStepMomentum', 'StepMomentum', 'ConstantParamScheduler',
    'CosineAnnealingParamScheduler', 'ExponentialParamScheduler',
    'LinearParamScheduler', 'MultiStepParamScheduler', 'StepParamScheduler',
    '_ParamScheduler', 'OptimWrapper', 'AmpOptimWrapper', 'ApexOptimWrapper',
    'OptimWrapperDict', 'OneCycleParamScheduler', 'OneCycleLR', 'PolyLR',
    'PolyMomentum', 'PolyParamScheduler', 'ReduceOnPlateauLR',
    'ReduceOnPlateauMomentum', 'ReduceOnPlateauParamScheduler',
    'ZeroRedundancyOptimizer', 'BaseOptimWrapper'
]
