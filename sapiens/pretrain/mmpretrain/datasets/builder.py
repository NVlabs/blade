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

from mmpretrain.registry import DATASETS


def build_dataset(cfg):
    """Build dataset.

    Examples:
        >>> from mmpretrain.datasets import build_dataset
        >>> mnist_train = build_dataset(
        ...     dict(type='MNIST', data_prefix='data/mnist/', test_mode=False))
        >>> print(mnist_train)
        Dataset MNIST
            Number of samples:  60000
            Number of categories:       10
            Prefix of data:     data/mnist/
        >>> mnist_test = build_dataset(
        ...     dict(type='MNIST', data_prefix='data/mnist/', test_mode=True))
        >>> print(mnist_test)
        Dataset MNIST
            Number of samples:  10000
            Number of categories:       10
            Prefix of data:     data/mnist/
    """
    return DATASETS.build(cfg)
