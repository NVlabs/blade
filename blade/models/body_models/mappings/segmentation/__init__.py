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
from mmhuman3d.core.conventions.segmentation import (body_segmentation as
                                                     body_segmentation_)
from .smpl_1723 import SMPL_SEGMENTATION_DICT_1723


class body_segmentation(body_segmentation_):
    new_super_set = {
        'LHAND': ['leftHand', 'leftHandIndex1'],
        'RHAND': ['rightHand', 'rightHandIndex1'],
        'TORSO':
        ['spine1', 'spine2', 'leftShoulder', 'rightShoulder', 'spine', 'hips']
    }

    def __init__(self, model_type='smpl', downsample=False) -> None:
        super().__init__(model_type)
        if downsample:
            self.DICT = SMPL_SEGMENTATION_DICT_1723

    def __getitem__(self, key):
        if key in self.new_super_set:
            result = []
            for k in self.new_super_set[key]:
                result += super().__getitem__(k)
            return result
        else:
            return super().__getitem__(key)
