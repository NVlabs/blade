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
import pytest
import torch
from torch.utils import model_zoo

from mmcv.utils import TORCH_VERSION, digit_version, load_url


@pytest.mark.skipif(
    torch.__version__ == 'parrots', reason='not necessary in parrots test')
def test_load_url():
    url1 = 'https://download.openmmlab.com/mmcv/test_data/saved_in_pt1.5.pth'
    url2 = 'https://download.openmmlab.com/mmcv/test_data/saved_in_pt1.6.pth'

    # The 1.6 release of PyTorch switched torch.save to use a new zipfile-based
    # file format. It will cause RuntimeError when a checkpoint was saved in
    # torch >= 1.6.0 but loaded in torch < 1.7.0.
    # More details at https://github.com/open-mmlab/mmpose/issues/904
    if digit_version(TORCH_VERSION) < digit_version('1.7.0'):
        model_zoo.load_url(url1)
        with pytest.raises(RuntimeError):
            model_zoo.load_url(url2)
    else:
        # high version of PyTorch can load checkpoints from url, regardless
        # of which version they were saved in
        model_zoo.load_url(url1)
        model_zoo.load_url(url2)

    load_url(url1)
    # if a checkpoint was saved in torch >= 1.6.0 but loaded in torch < 1.5.0,
    # it will raise a RuntimeError
    if digit_version(TORCH_VERSION) < digit_version('1.5.0'):
        with pytest.raises(RuntimeError):
            load_url(url2)
    else:
        load_url(url2)
