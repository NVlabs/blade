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
import glob
import os

import torch

if torch.__version__ == 'parrots':
    import parrots

    def get_compiler_version():
        return 'GCC ' + parrots.version.compiler

    def get_compiling_cuda_version():
        return parrots.version.cuda
else:
    from ..utils import ext_loader
    ext_module = ext_loader.load_ext(
        '_ext', ['get_compiler_version', 'get_compiling_cuda_version'])

    def get_compiler_version():
        return ext_module.get_compiler_version()

    def get_compiling_cuda_version():
        return ext_module.get_compiling_cuda_version()


def get_onnxruntime_op_path():
    wildcard = os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
        '_ext_ort.*.so')

    paths = glob.glob(wildcard)
    if len(paths) > 0:
        return paths[0]
    else:
        return ''
