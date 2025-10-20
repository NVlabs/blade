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

import gzip
import io
import pickle

import cv2
import numpy as np


def datafrombytes(content: bytes, backend: str = 'numpy') -> np.ndarray:
    """Data decoding from bytes.

    Args:
        content (bytes): The data bytes got from files or other streams.
        backend (str): The data decoding backend type. Options are 'numpy',
            'nifti', 'cv2' and 'pickle'. Defaults to 'numpy'.

    Returns:
        numpy.ndarray: Loaded data array.
    """
    if backend == 'pickle':
        data = pickle.loads(content)
    else:
        with io.BytesIO(content) as f:
            if backend == 'nifti':
                f = gzip.open(f)
                try:
                    from nibabel import FileHolder, Nifti1Image
                except ImportError:
                    print('nifti files io depends on nibabel, please run'
                          '`pip install nibabel` to install it')
                fh = FileHolder(fileobj=f)
                data = Nifti1Image.from_file_map({'header': fh, 'image': fh})
                data = Nifti1Image.from_bytes(data.to_bytes()).get_fdata()
            elif backend == 'numpy':
                data = np.load(f)
            elif backend == 'cv2':
                data = np.frombuffer(f.read(), dtype=np.uint8)
                data = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            else:
                raise ValueError
    return data
