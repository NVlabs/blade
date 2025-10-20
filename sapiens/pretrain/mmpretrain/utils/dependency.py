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

import re
from functools import wraps
from inspect import isfunction

from importlib_metadata import PackageNotFoundError, distribution
from mmengine.utils import digit_version


def satisfy_requirement(dep):
    pat = '(' + '|'.join(['>=', '==', '>']) + ')'
    parts = re.split(pat, dep, maxsplit=1)
    parts = [p.strip() for p in parts]
    package = parts[0]
    if len(parts) > 1:
        op, version = parts[1:]
        op = {
            '>=': '__ge__',
            '==': '__eq__',
            '>': '__gt__',
            '<': '__lt__',
            '<=': '__le__'
        }[op]
    else:
        op, version = None, None

    try:
        dist = distribution(package)
        if op is None or getattr(digit_version(dist.version), op)(
                digit_version(version)):
            return True
    except PackageNotFoundError:
        pass

    return False


def require(dep, install=None):
    """A wrapper of function for extra package requirements.

    Args:
        dep (str): The dependency package name, like ``transformers``
            or ``transformers>=4.28.0``.
        install (str, optional): The installation command hint. Defaults
            to None, which means to use "pip install dep".
    """

    def wrapper(fn):
        assert isfunction(fn)

        @wraps(fn)
        def ask_install(*args, **kwargs):
            name = fn.__qualname__.replace('.__init__', '')
            ins = install or f'pip install "{dep}"'
            raise ImportError(
                f'{name} requires {dep}, please install it by `{ins}`.')

        if satisfy_requirement(dep):
            fn._verify_require = getattr(fn, '_verify_require', lambda: None)
            return fn

        ask_install._verify_require = ask_install
        return ask_install

    return wrapper


WITH_MULTIMODAL = all(
    satisfy_requirement(item)
    for item in ['pycocotools', 'transformers>=4.28.0'])


def register_multimodal_placeholder(names, registry):
    for name in names:

        def ask_install(*args, **kwargs):
            raise ImportError(
                f'{name} requires extra multi-modal dependencies, please '
                'install it by `pip install "mmpretrain[multimodal]"` '
                'or `pip install -e ".[multimodal]"`.')

        registry.register_module(name=name, module=ask_install)
