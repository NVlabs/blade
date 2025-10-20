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
import numpy as np
import pytest

import mmcv


def test_color():
    assert mmcv.color_val(mmcv.Color.blue) == (255, 0, 0)
    assert mmcv.color_val('green') == (0, 255, 0)
    assert mmcv.color_val((1, 2, 3)) == (1, 2, 3)
    assert mmcv.color_val(100) == (100, 100, 100)
    assert mmcv.color_val(np.zeros(3, dtype=int)) == (0, 0, 0)
    with pytest.raises(TypeError):
        mmcv.color_val([255, 255, 255])
    with pytest.raises(TypeError):
        mmcv.color_val(1.0)
    with pytest.raises(AssertionError):
        mmcv.color_val((0, 0, 500))
