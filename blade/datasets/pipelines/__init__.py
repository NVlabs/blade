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
from .transforms import (RandomChannelNoise, RandomHorizontalFlip, Rotation,
                         CenterCrop, MeshAffine, Normalize, ColorJitter,
                         Lighting, GenHeatmap, ResizeSample, GetBboxInfo,
                         GetRandomScaleRotation)

from .loading import LoadUVDfromFile, LoadSegfromFile
from mmhuman3d.data.datasets.builder import PIPELINES

PIPELINES.register_module(name=['GetRandomScaleRotation'],
                          module=GetRandomScaleRotation,
                          force=True)
PIPELINES.register_module(name=['RandomChannelNoise'],
                          module=RandomChannelNoise,
                          force=True)
PIPELINES.register_module(name=['RandomHorizontalFlip'],
                          module=RandomHorizontalFlip,
                          force=True)
PIPELINES.register_module(name=['ResizeSample'],
                          module=ResizeSample,
                          force=True)
PIPELINES.register_module(name=['Rotation'], module=Rotation, force=True)
PIPELINES.register_module(name=['CenterCrop'], module=CenterCrop, force=True)
PIPELINES.register_module(name=['MeshAffine'], module=MeshAffine, force=True)
PIPELINES.register_module(name=['Normalize'], module=Normalize, force=True)
PIPELINES.register_module(name=['ColorJitter'], module=ColorJitter, force=True)
PIPELINES.register_module(name=['Lighting'], module=Lighting, force=True)
PIPELINES.register_module(name=['GenHeatmap'], module=GenHeatmap, force=True)
PIPELINES.register_module(name=['LoadUVDfromFile'],
                          module=LoadUVDfromFile,
                          force=True)

PIPELINES.register_module(name=['LoadSegfromFile'],
                          module=LoadSegfromFile,
                          force=True)
PIPELINES.register_module(name=['GetBboxInfo'], module=GetBboxInfo, force=True)
