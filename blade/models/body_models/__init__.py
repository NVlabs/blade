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
from .mano import MANO, MANO_LEFT, MANO_RIGHT
from .smpl import SMPL
from .smpl_d import SMPL_D, SMPLH_D
from .smplh import SMPLH
from .smpl_x import SMPLX
from .flame import FLAME

__all__ = [
    'MANO', 'SMPL', 'SMPLH', 'SMPLX', 'FLAME', 'SMPLD', 'MANO_LEFT',
    'MANO_RIGHT'
]


def get_model_class(key):
    key = key.lower().replace('-', '')
    model_class = dict(smpl=SMPL,
                       mano=MANO,
                       flame=FLAME,
                       smplh=SMPLH,
                       smplx=SMPLX,
                       smpl_d=SMPL_D,
                       smplh_d=SMPLH_D,
                       mano_left=MANO_LEFT,
                       mano_right=MANO_RIGHT)
    return model_class[key]
