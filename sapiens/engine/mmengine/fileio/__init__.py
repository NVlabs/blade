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

from .backends import (BaseStorageBackend, HTTPBackend, LmdbBackend,
                       LocalBackend, MemcachedBackend, PetrelBackend,
                       register_backend)
from .file_client import FileClient, HardDiskBackend
from .handlers import (BaseFileHandler, JsonHandler, PickleHandler,
                       YamlHandler, register_handler)
from .io import (copy_if_symlink_fails, copyfile, copyfile_from_local,
                 copyfile_to_local, copytree, copytree_from_local,
                 copytree_to_local, dump, exists, generate_presigned_url, get,
                 get_file_backend, get_local_path, get_text, isdir, isfile,
                 join_path, list_dir_or_file, load, put, put_text, remove,
                 rmtree)
from .parse import dict_from_file, list_from_file

__all__ = [
    'BaseStorageBackend', 'FileClient', 'PetrelBackend', 'MemcachedBackend',
    'LmdbBackend', 'HardDiskBackend', 'LocalBackend', 'HTTPBackend',
    'copy_if_symlink_fails', 'copyfile', 'copyfile_from_local',
    'copyfile_to_local', 'copytree', 'copytree_from_local',
    'copytree_to_local', 'exists', 'generate_presigned_url', 'get',
    'get_file_backend', 'get_local_path', 'get_text', 'isdir', 'isfile',
    'join_path', 'list_dir_or_file', 'put', 'put_text', 'remove', 'rmtree',
    'load', 'dump', 'register_handler', 'BaseFileHandler', 'JsonHandler',
    'PickleHandler', 'YamlHandler', 'list_from_file', 'dict_from_file',
    'register_backend'
]
