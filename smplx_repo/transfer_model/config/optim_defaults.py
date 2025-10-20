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
# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: Vassilis Choutas, vassilis.choutas@tuebingen.mpg.de

from typing import Tuple
from omegaconf import OmegaConf
from dataclasses import dataclass


@dataclass
class LBFGS:
    line_search_fn: str = 'strong_wolfe'
    max_iter: int = 50


@dataclass
class SGD:
    momentum: float = 0.9
    nesterov: bool = True


@dataclass
class ADAM:
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    amsgrad: bool = False


@dataclass
class RMSProp:
    alpha: float = 0.99


@dataclass
class TrustRegionNewtonCG:
    max_trust_radius: float = 1000
    initial_trust_radius: float = 0.05
    eta: float = 0.15
    gtol: float = 1e-05


@dataclass
class OptimConfig:
    type: str = 'trust-ncg'
    lr: float = 1.0
    gtol: float = 1e-8
    ftol: float = -1.0
    maxiters: int = 100

    lbfgs: LBFGS = LBFGS()
    sgd: SGD = SGD()
    adam: ADAM = ADAM()
    trust_ncg: TrustRegionNewtonCG = TrustRegionNewtonCG()


conf = OmegaConf.structured(OptimConfig)
