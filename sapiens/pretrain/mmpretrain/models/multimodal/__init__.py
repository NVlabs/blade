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

from mmpretrain.utils.dependency import WITH_MULTIMODAL

if WITH_MULTIMODAL:
    from .blip import *  # noqa: F401,F403
    from .blip2 import *  # noqa: F401,F403
    from .chinese_clip import *  # noqa: F401, F403
    from .clip import *  # noqa: F401, F403
    from .flamingo import *  # noqa: F401, F403
    from .llava import *  # noqa: F401, F403
    from .minigpt4 import *  # noqa: F401, F403
    from .ofa import *  # noqa: F401, F403
    from .otter import *  # noqa: F401, F403
else:
    from mmpretrain.registry import MODELS
    from mmpretrain.utils.dependency import register_multimodal_placeholder

    register_multimodal_placeholder([
        'Blip2Caption', 'Blip2Retrieval', 'Blip2VQA', 'BlipCaption',
        'BlipNLVR', 'BlipRetrieval', 'BlipGrounding', 'BlipVQA', 'Flamingo',
        'OFA', 'ChineseCLIP', 'MiniGPT4', 'Llava', 'Otter', 'CLIP',
        'CLIPZeroShot'
    ], MODELS)
