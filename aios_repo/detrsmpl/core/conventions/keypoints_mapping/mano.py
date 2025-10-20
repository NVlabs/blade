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
# Original order from MANO J_regressor
MANO_RIGHT_KEYPOINTS = [
    'right_wrist', 'right_index_1', 'right_index_2', 'right_index_3',
    'right_middle_1', 'right_middle_2', 'right_middle_3', 'right_pinky_1',
    'right_pinky_2', 'right_pinky_3', 'right_ring_1', 'right_ring_2',
    'right_ring_3', 'right_thumb_1', 'right_thumb_2', 'right_thumb_3',
    'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky'
]

MANO_LEFT_KEYPOINTS = [
    x.replace('right_', 'left_') for x in MANO_RIGHT_KEYPOINTS
]

# Re-arranged order is compatible with the output of manolayer
# from official [manopth](https://github.com/hassony2/manopth)
MANO_REORDER_MAP = [
    0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20
]

MANO_RIGHT_REORDER_KEYPOINTS = [
    MANO_RIGHT_KEYPOINTS[i] for i in MANO_REORDER_MAP
]
MANO_LEFT_REORDER_KEYPOINTS = [
    MANO_LEFT_KEYPOINTS[i] for i in MANO_REORDER_MAP
]

# Deprecated: reserved for backward compatibility
MANO_KEYPOINTS = MANO_RIGHT_KEYPOINTS
# Two hands (left + right)
MANO_HANDS_KEYPOINTS = MANO_LEFT_KEYPOINTS + MANO_RIGHT_KEYPOINTS
# Reordered two hands (left + right)
MANO_HANDS_REORDER_KEYPOINTS = \
        MANO_LEFT_REORDER_KEYPOINTS + MANO_RIGHT_REORDER_KEYPOINTS
