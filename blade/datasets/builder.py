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



from .human_image_dataset import HumanImageDataset
from .human_image_dataset_tz import HumanImageDataset_Tz
from .human_image_dataset_smplx import HumanImageDataset_SMPLX
from .our_dataset import OurDataset
from .our_dataset_smplx import OurDataset_SMPLX
from mmhuman3d.data.datasets.builder import build_dataset, DATASETS, build_dataloader
# from .demo_dataset import DemoDataset
from .itw_dataset import ITWDataset

# DATASETS.register_module(name=['DemoDataset'], module=DemoDataset)
# DATASETS.register_module(name=['ImageDataset'], module=ImageDataset)
DATASETS.register_module(name=['HumanImageDataset'],
                         module=HumanImageDataset,
                         force=True)
DATASETS.register_module(name=['HumanImageDataset_Tz'],
                         module=HumanImageDataset_Tz,
                         force=True)
DATASETS.register_module(name=['HumanImageDataset_SMPLX'],
                         module=HumanImageDataset_SMPLX,
                         force=True)
DATASETS.register_module(name=['OurDataset'], module=OurDataset, force=True)
DATASETS.register_module(name=['OurDataset_SMPLX'], module=OurDataset_SMPLX, force=True)
DATASETS.register_module(name=['ITWDataset'], module=ITWDataset, force=True)

__all__ = ['build_dataset', 'build_dataloader']
