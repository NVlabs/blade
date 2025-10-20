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


import os.path as osp, pytorch3d as p3d, shutil, random, argparse, warnings, traceback
from argparse import ArgumentParser
import torch, math, mmcv, urllib, numpy as np, itertools, torch.nn.functional as F, sys, torch.nn as nn, os, sys, wandb
import torchvision.transforms.functional as TF
from torchvision.ops import masks_to_boxes
from torch.cuda.amp import autocast
from typing import Optional, Tuple, Union
from types import MethodType
import numpy as np
from pytorch3d.transforms import so3_relative_angle, axis_angle_to_matrix
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
from os.path import abspath, dirname
from collections import OrderedDict

from mmhuman3d.utils.geometry import batch_rodrigues
from mmhuman3d.core.conventions.segmentation import body_segmentation
from mmhuman3d.models.architectures.base_architecture import BaseArchitecture
from blade.utils.torch_utils import dict2numpy
from blade.structures.meshes.utils import MeshSampler
from blade.cameras.builder import build_cameras
from blade.models.body_models.mappings import get_keypoint_idx, convert_kps
from blade.models.body_models.builder import build_body_model
from blade.models.losses.builder import build_loss
from blade.models.heads.builder import build_head
from blade.render.builder import build_renderer
from blade.cameras.utils import (pred_cam_to_transl,
                                 estimate_cam_weakperspective_batch,
                                 project_points_pred_cam,
                                 project_points_focal_length_pixel)
from blade.render.explicit.pytorch3d_wrapper.textures.builder import (
    build_textures)

root_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.append(root_dir)
from blade.utils.helpers import resize

# DAV2
from DAv2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

# AiOS
sys.path.append(osp.join(root_dir, 'aios_repo'))

from blade.utils.helpers import *

# convert SMPL-X -> SMPL
import smplx
from pytorch3d.transforms import matrix_to_axis_angle
sys.path.insert(0, os.path.abspath(os.path.join(root_dir, 'smplx_repo')))
from transfer_model.transfer_model import run_fitting
from blade.datasets.utils import init_conversion

# pose estimator & pytorch3d solver for (f, Tx, Ty)
from blade.models.architectures.blade_helper import *
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.apis import init_model as init_pose_estimator
from mmpose.apis import inference_topdown
from mmseg.apis import inference_model, init_model
from mmpose.evaluation.functional import nms
from smplx_repo.smplx.joint_names import JOINT_NAMES as smplx_joint_names
import mediapipe as mp
import cv2
import torch.optim as optim
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer, SoftPhongShader, DirectionalLights,
    MeshRasterizer,
    SoftSilhouetteShader, # SoftPhongShader
    TexturesVertex,
    BlendParams
)

import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')