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


import argparse, socket, time, sys, math
from PIL import Image
import torch, numpy as np, os, time
from os.path import abspath, dirname
# from blade.configs.base import root
from glob import glob

from gitdb.fun import chunk_size

# from blade.models.body_models.smpl_x import SMPLX
from blade.models.body_models.builder import build_body_model
from blade.configs.base import body_model_train
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../smplx_repo')))
from transfer_model.optimizers import build_optimizer, minimize
from transfer_model.transfer_model import run_fitting
import smplx
from pytorch3d.transforms import euler_angles_to_matrix, axis_angle_to_matrix, matrix_to_euler_angles, matrix_to_axis_angle
from utils import init_conversion
from gender_lookup import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROCESS_ID = int(sys.argv[1])
SEQ_START = int(sys.argv[2])
SEQ_END = int(sys.argv[3])


DO_VIS = False
DO_WRITE_SMPLX = True

repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
data_root = os.path.join(repo_root, 'mmhuman_data')

frame_offset = 1
seq_folders = sorted(glob(f'{data_root}/datasets/bedlamcc/png/seq_*'))
output_fn = f'{data_root}/preprocessed_datasets/bedlamcc_smplx_chunks/bedlamcc_{PROCESS_ID}_{SEQ_START}_{SEQ_END}.npz'

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')
# device = torch.device('cuda')
device = torch.device('cpu')

model_path = f'{dirname(dirname(dirname(abspath(__file__))))}/body_models/'
smplx_model_female = smplx.create(model_path, model_type='smplx', gender='female', use_pca=False, flat_hand_mean=True).to(device)
smplx_model_male = smplx.create(model_path, model_type='smplx', gender='male', use_pca=False, flat_hand_mean=True).to(device)
smplx_model_neutral = smplx.create(model_path, model_type='smplx', gender='neutral', use_pca=False, flat_hand_mean=True).to(device)
verify_smpl_model = build_body_model(body_model_train).to(device)

smpl_repo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '../smplx_repo')
conversion_initialized = False

out = {'img_fn': [], 'has_keypoints2d': [],
       'sample_idx': [], 'gender': [],
        'smplx':{
            'body_pose': [], 'left_hand_pose': [], 'right_hand_pose': [],
            'global_orient': [], 'betas': [], 'transl': [],
           },
       'focal_length': [], 'K': [], 'ori_shape': [], 'center': [], 'scale': []}

total_cnt_i = 0
seq_cnt_i = 0
t0 = time.time()
# chunk_size = int(len(seq_folders) / N_TOTAL_CHUNKS + 1)
# seqs_to_process = seq_folders[CUR_CHUNK * chunk_size : (CUR_CHUNK+1) * chunk_size]

seqs_to_process = seq_folders[SEQ_START : SEQ_END]
for seq in seqs_to_process:
    seq_cnt_i += 1
    seq_i = int(seq.split('seq_')[-1])
    gt_fn = f'{seq.replace("png", "info")}/camera_smplx.npz'
    gt = np.load(gt_fn, allow_pickle=True)

    cnt_i = 0
    for frame_i in range(1000):
        cnt_i += 1
        print(f'\n\nchunk {PROCESS_ID}, seq {seq_i} frame {cnt_i}, {time.time()-t0} seconds so far')

        img_fn = f'{seq}/seq_{seq_i:06d}_{frame_i:04d}.png'
        if not os.path.exists(img_fn):
            img_fn = f'{seq}/seq_{seq_i:06d}_FinalImage_{frame_i:04d}.png'
        if not os.path.exists(img_fn):
            continue

        #  load images
        im = Image.open(img_fn).convert('RGB')
        # plt.imshow(im); plt.show()
        h, w = im.height, im.width

        # load data
        cam_pose = torch.tensor(gt['cam_pose'][frame_i]).to(device)
        smplx_betas = torch.tensor(gt['smplx_betas'])[None].float().to(device)
        smplx_poses = torch.tensor(gt['smplx_poses'][frame_i + frame_offset][3:]).to(device)
        smplx_global_ori = torch.tensor(gt['smplx_global_ori'][frame_i + frame_offset])[None].float().to(device)
        smplx_trans = torch.tensor(gt['smplx_trans'][frame_i + frame_offset])[None].float().to(device)

        smplx_body = torch.tensor(smplx_poses[:63])[None].float().to(device)
        # smplx_hand_l = torch.tensor(smplx_poses[63:(63 + 45)])[None].float().to(device)
        # smplx_hand_r = torch.tensor(smplx_poses[(63 + 45):(63 + 45*2)])[None].float().to(device)
        smplx_hand_l = torch.tensor(smplx_poses[72:117])[None].float().to(device)
        smplx_hand_r = torch.tensor(smplx_poses[117:])[None].float().to(device)
        # smplx_jaw = torch.tensor(smplx_poses[63+90:63+90+3])[None].float()
        smplx_eye_l = torch.tensor(smplx_poses[(63 + 45*2 + 3):(63 + 45*2 + 3*2)])[None].float().to(device)
        smplx_eye_r = torch.tensor(smplx_poses[63 + 90 + 3 + 3:])[None].float().to(device)

        # betas = torch.zeros(1, 10)  # Shape parameters
        # global_orient = torch.zeros(1, 3)  # Global orientation
        # body_pose = torch.zeros(1, 63)  # Body pose
        expression = torch.zeros(1, 10).to(device)  # Expression parameters
        smplx_jaw = torch.zeros(1, 3).to(device)  # Expression parameters
        # transl = torch.zeros(1, 3)  # Translation

        # get camera pose
        cam_pose = gt['cam_pose'][frame_i]
        x, y, z, yaw, pitch, roll = cam_pose[0], cam_pose[1], cam_pose[2], cam_pose[3], cam_pose[4], cam_pose[5]
        yaw = yaw - 180  # michael had to add an additional flipping, here we remove it

        # -------------- ver 4 convention
        # roll = roll - 180  # michael had to add an additional flipping, here we remove it

        yaw = torch.deg2rad(torch.tensor(yaw))
        pitch = torch.deg2rad(torch.tensor(pitch))
        roll = torch.deg2rad(torch.tensor(roll))

        # UE to OpenCV
        T_ue_to_cv = torch.tensor([[0, 1, 0],
                                   [0, 0, -1],
                                   [1, 0, 0]]).float().to(device)

        cam_orientation_ue = euler_angles_to_matrix(torch.tensor([yaw, pitch, roll]).float(), 'ZYX').to(device)
        cam_orientation_cv = T_ue_to_cv @ cam_orientation_ue @ T_ue_to_cv.t()
        cam_position = T_ue_to_cv @ torch.tensor([x, y, z])[:, None].to(device)

        # CV world -> camera
        # -------------- ver 3 convension
        # T_flip = torch.eye(3).float().to(device)

        # -------------- ver 4 convention
        T_flip = torch.tensor([[-1, 0, 0],
                               [0, 1, 0],
                               [0, 0, -1]]).float().to(device)

        R = T_flip @ cam_orientation_cv.t()
        t = - T_flip @ cam_orientation_cv.t() @ cam_position / 100.

        # SMPL -> world
        # T_smpl2cv = torch.tensor([[1, 0, 0],
        #                           [0, -1, 0],
        #                           [0, 0, -1]]).float()
        T_smpl2cv = torch.tensor([[0, 0, 1],
                                  [0, -1, 0],
                                  [1, 0, 0]]).float().to(device)  # to CV then yaw 90 degrees to left

        actor_name = gt['subject_id'].item()
        if any(name in actor_name for name in males):
            smplx_model = smplx_model_male
            gender = 'male'
        elif any(name in actor_name for name in females):
            smplx_model = smplx_model_female
            gender = 'female'
        else:
            smplx_model = smplx_model_neutral
            gender = 'neutral'

        # identity_rot = matrix_to_axis_angle(torch.eye(3))[None]
        output = smplx_model(betas=smplx_betas, body_pose=smplx_body, left_hand_pose=smplx_hand_l,
                             right_hand_pose=smplx_hand_r,
                             jaw_pose=smplx_jaw, leye_pose=smplx_eye_l, reye_pose=smplx_eye_r,
                             global_orient=torch.zeros((1, 3)), transl=torch.zeros((1, 3)),
                             expression=expression, return_verts=True)
        vertices = output.vertices
        pelvis = output.joints[0, 0, :, None]

        # intrinsics
        sensor_size = 36
        f = gt['focal_length'][frame_i] / sensor_size * w
        K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]).float().to(device)

        global_ori_mat = axis_angle_to_matrix(smplx_global_ori)[0]
        R_tmp = R @ T_smpl2cv

        transl_cam = R_tmp @ (- global_ori_mat @ pelvis + pelvis + smplx_trans.t()) + t
        rot_cam = R_tmp @ global_ori_mat
        vert_cam = rot_cam @ vertices[0].t() + transl_cam

        # if vert_cam[-1].min(-1)[0] < 0.2:

        vert_2D = K @ vert_cam
        vert_2D[0] /= vert_2D[2]
        vert_2D[1] /= vert_2D[2]

        x_min, y_min, _ = vert_2D.min(axis=1)[0].clamp(min=0)
        x_max, y_max, _ = vert_2D.max(axis=1)[0]
        x_max = x_max.clamp(max=w)
        y_max = y_max.clamp(max=h)

        # Calculate the center of the bounding box
        center_x = ((x_min + x_max) / 2).clamp(min=0, max=w)
        center_y = ((y_min + y_max) / 2).clamp(min=0, max=h)
        center = torch.stack([center_x, center_y], axis=0).numpy()
        out['center'].append(center)

        # Calculate the width and height of the bounding box
        bbox_expand_scale = 1.25
        scale = min(max((x_max - x_min), (y_max - y_min)), max(h, w)) * bbox_expand_scale
        out['scale'].append(scale)

        # NOTE: global ori rotates around pelvis, but it's not at origin,
        #  so pelvis need to first get the old translation including pelvis (T_old)
        #  then add that translation (T_old) to the new model after subtracting the new pelvis
        #   R_tmp @ (global_ori_mat @ (vertices[0].t() - pelvis) + pelvis + smplx_trans.t()) + t
        output = smplx_model(betas=smplx_betas, body_pose=smplx_body, left_hand_pose=smplx_hand_l,
                             right_hand_pose=smplx_hand_r,
                             jaw_pose=smplx_jaw, leye_pose=smplx_eye_l, reye_pose=smplx_eye_r,
                             global_orient=matrix_to_axis_angle(R_tmp @ global_ori_mat)[None],
                             transl=torch.zeros((1, 3)),
                             expression=expression, return_verts=True)
        # vertices = output.vertices
        pelvis_new = output.joints[0, 0, :, None]
        final_ori = matrix_to_axis_angle(R_tmp @ global_ori_mat)[None]
        final_transl = (- pelvis_new + R_tmp @ (pelvis + smplx_trans.t()) + t).t()

        out['gender'].append(gender)
        # out['sequence_name'].append(img_fn.split('/')[-1])
        # out['image_name'].append(img_fn.split('/')[-1])
        out['img_fn'].append(img_fn.split('/')[-2] + '/' + img_fn.split('/')[-1])
        out['sample_idx'].append(total_cnt_i)
        out['has_keypoints2d'].append(0)
        out['smplx']['betas'].append(smplx_betas)
        out['smplx']['body_pose'].append(smplx_body)
        out['smplx']['left_hand_pose'].append(smplx_hand_l)
        out['smplx']['right_hand_pose'].append(smplx_hand_r)
        out['smplx']['global_orient'].append(final_ori)
        out['smplx']['transl'].append(final_transl)

        out['focal_length'].append(f)
        out['K'].append(K)
        out['ori_shape'].append((h, w))

        total_cnt_i += 1

        #  render SMPLX & SMPL
        if DO_VIS:
            output = smplx_model(betas=smplx_betas, body_pose=smplx_body, left_hand_pose=smplx_hand_l,
                                 right_hand_pose=smplx_hand_r,
                                 jaw_pose=smplx_jaw, leye_pose=smplx_eye_l, reye_pose=smplx_eye_r,
                                 global_orient=final_ori,
                                 transl=final_transl,
                                 expression=expression, return_verts=True)
            vertices = output.vertices
            pelvis_new = output.joints[0, 0, :, None]

            transl_cam_new = - pelvis_new + R_tmp @ (pelvis + smplx_trans.t()) + t

            vert_2D = K @ (vertices[0].t() + transl_cam_new)
            vert_2D[0] /= vert_2D[2]
            vert_2D[1] /= vert_2D[2]

            cx, cy = center
            width = height = scale
            top_left_x = cx - width / 2
            top_left_y = cy - height / 2
            _, ax = plt.subplots()
            rect = patches.Rectangle((top_left_x, top_left_y), width, height, linewidth=2, edgecolor='r',
                                     facecolor='none')
            # Add the rectangle to the plot
            ax.add_patch(rect)
            ax.plot(cx, cy, "bo")  # Mark the center for reference
            plt.imshow(im);
            plt.scatter(vert_2D[0], vert_2D[1], s=0.2);
            ax.set_xlim(0, w)  # Adjust as needed for your image
            ax.set_ylim(0, h)
            plt.gca().invert_yaxis()
            plt.show()
            time.sleep(0.1)

t1 = time.time()

torch.save(out, output_fn)
print(f'total time: {t1-t0}, saved to {output_fn}')
