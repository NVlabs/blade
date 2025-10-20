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


import mmcv, os, torch, copy, torch.nn.functional as F, random, datetime
import numpy as np
from torchvision import transforms
from typing import Optional, Union, List
from collections import OrderedDict
from mmhuman3d.data.datasets.human_image_dataset import HumanImageDataset as _HumanImageDataset
from blade.datasets.pipelines.transforms import _rotate_smpl_pose, _flip_smplx_pose, _flip_axis_angle, _flip_hand_pose

from blade.models.body_models.mappings import get_keypoint_idx
from blade.utils.bbox_utils import kp2d_to_bbox
from blade.datasets.pipelines.compose import Compose

import matplotlib.pyplot as plt


def discard_alpha_and_scale(x):
    return 255.0 * x[:3]  # Discard alpha component and scale by 255


def make_depth_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        discard_alpha_and_scale,
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])


class HumanImageDataset_SMPLX(_HumanImageDataset):
    ALLOWED_METRICS = {
        'mpjpe', 'pa-mpjpe', 'pve', '3dpck', 'pa-3dpck', '3dauc', 'pa-3dauc',
        'ihmr', 'pa-pve', 'raw_miou', 'raw_pmiou', 'opti_miou', 'opti_pmiou',
        'opti_miou_w_gt_mask', 'opti_pmiou_w_gt_mask', 'z_error', 'distortion_error',
        'inv_z_error', 'xy_error', 'f_perc_error'
    }

    def __init__(
        self,
        data_prefix: str,
        pipeline: list,
        dataset_name: str,
        body_model: Optional[Union[dict, None]] = None,
        ann_file: Optional[Union[str, None]] = None,
        convention: Optional[str] = 'human_data',
        cache_data_path: Optional[Union[str, None]] = None,
        test_mode: Optional[bool] = False,
        is_distorted: Optional[bool] = False,  # new feature
        num_data: Optional[int] = None,  # new feature
        start_index: int = None,
        do_black_pad_aug = True
    ):
        super().__init__(data_prefix, pipeline, dataset_name, body_model,
                         ann_file, convention, cache_data_path, test_mode)
        self.start_index = start_index
        if start_index is not None:
            self.num_data = self.num_data - start_index
        if num_data is not None:
            self.num_data = num_data
        self.is_distorted = is_distorted

        self.pipeline_load = Compose(pipeline[:1])

        Collect_idx = None
        for i in range(1, len(pipeline)):
            if pipeline[i]['type'] == 'Collect':
                Collect_idx = i+1
                break

        self.pipeline_depth = Compose(pipeline[1:Collect_idx])
        self.pipeline_pose = Compose(pipeline[Collect_idx:])

        self.do_black_pad_aug = do_black_pad_aug

    def prepare_raw_data(self, idx: int):
        """Get item from self.human_data."""
        if self.start_index is not None:
            idx = idx + self.start_index
        sample_idx = idx

        if self.cache_reader is not None:
            self.human_data = self.cache_reader.get_item(idx)
            idx = idx % self.cache_reader.slice_size
        info = {}
        info['img_prefix'] = None
        image_path = self.human_data['image_path'][idx]
        if self.dataset_name == 'h36m':
            image_path = image_path.replace('/images/', '/')

        info['image_path'] = os.path.join(self.data_prefix, 'datasets',
                                          self.dataset_name, image_path)

        if image_path.endswith('smc'):
            device, device_id, frame_id = self.human_data['image_id'][idx]
            info['image_id'] = (device, int(device_id), int(frame_id))

        info['dataset_name'] = self.dataset_name
        info['sample_idx'] = sample_idx

        if 'keypoints2d' in self.human_data:
            info['keypoints2d'] = self.human_data['keypoints2d'][idx]
            info['has_keypoints2d'] = 1
        else:
            info['keypoints2d'] = np.zeros((self.num_keypoints, 3))
            info['has_keypoints2d'] = 0

        if 'bbox_xywh' in self.human_data:
            info['bbox_xywh'] = self.human_data['bbox_xywh'][idx]
            x, y, w, h, _ = info['bbox_xywh']
            cx = x + w / 2
            cy = y + h / 2
            w = h = max(w, h)
            info['center'] = np.array([cx, cy])
            info['scale'] = np.array([w, h])
        elif 'keypoints2d' in self.human_data:
            bbox_xywh = kp2d_to_bbox(info['keypoints2d'],
                                     scale_factor=1.25,
                                     xywh=True)
            cx, cy, w, h = bbox_xywh[0]
            w = h = max(w, h)
            info['center'] = np.array([cx, cy])
            info['scale'] = np.array([w, h])
        else:
            info['bbox_xywh'] = np.zeros((5))
            info['center'] = np.zeros((2))
            info['scale'] = np.zeros((2))

        # ------------------- SMPL------------------------
        if 'smpl' in self.human_data:
            smpl_dict = self.human_data['smpl']
        else:
            smpl_dict = {}

        if 'smpl' in self.human_data:
            if 'has_smpl' in self.human_data:
                info['has_smpl'] = int(self.human_data['has_smpl'][idx])
            else:
                info['has_smpl'] = 1
        else:
            info['has_smpl'] = 0
        if 'body_pose' in smpl_dict:
            info['smpl_body_pose'] = smpl_dict['body_pose'][idx]
        else:
            info['smpl_body_pose'] = np.zeros((23, 3))

        if 'global_orient' in smpl_dict:
            info['smpl_global_orient'] = smpl_dict['global_orient'][idx]
        else:
            info['smpl_global_orient'] = np.zeros((3))

        info['smpl_origin_orient'] = info['smpl_global_orient'].copy()

        if 'betas' in smpl_dict:
            info['smpl_betas'] = smpl_dict['betas'][idx]
        else:
            info['smpl_betas'] = np.zeros((10))

        if 'betas_neutral' in smpl_dict:
            if not self.test_mode:  #for pw3d training
                info['smpl_betas'] = smpl_dict['betas_neutral'][idx]

        if 'transl' in smpl_dict:
            info['smpl_transl'] = smpl_dict['transl'][idx].astype(np.float32)
            info['has_transl'] = 1
        else:
            info['smpl_transl'] = np.zeros((3)).astype(np.float32)
            info['has_transl'] = 0

        # ------------------- SMPL------------------------
        if 'smplx' in self.human_data:
            SMPLX_dict = self.human_data['smplx']

            info['has_SMPLX'] = 1
            info['SMPLX_body_pose'] = SMPLX_dict['body_pose'][idx].detach().cpu().numpy()[0]
            info['SMPLX_left_hand_pose'] = SMPLX_dict['left_hand_pose'][idx].detach().cpu().numpy()[0]
            info['SMPLX_right_hand_pose'] = SMPLX_dict['right_hand_pose'][idx].detach().cpu().numpy()[0]
            info['SMPLX_global_orient'] = SMPLX_dict['global_orient'][idx].detach().cpu().numpy()[0]
            info['SMPLX_origin_orient'] = info['SMPLX_global_orient'].copy()[0]
            info['SMPLX_betas'] = SMPLX_dict['betas'][idx].detach().cpu().numpy()[0]
            info['SMPLX_transl'] = SMPLX_dict['transl'][idx].detach().cpu().numpy().astype(np.float32)[0]
        else:
            info['has_SMPLX'] = 0

        # ------------ others ----------------
        if 'focal_length' in self.human_data:
            info['ori_focal_length'] = float(
                self.human_data['focal_length'][idx].reshape(-1)[0])
            info['has_focal_length'] = 1
        else:
            # info['ori_focal_length'] = 5000.
            info['has_focal_length'] = 0

        if 'K' in self.human_data:
            info['K'] = self.human_data['K'][idx].reshape(3,
                                                          3).astype(np.float32)
            info['has_K'] = 1
        else:
            # info['K'] = np.eye(3, 3).astype(np.float32)
            info['has_K'] = 0

        if self.is_distorted:
            info['distortion_max'] = float(
                self.human_data['distortion_max'][idx])
            info['is_distorted'] = 1
        else:
            info['distortion_max'] = 1.0
            info['is_distorted'] = 0

        return info

    def prepare_data(self, idx: int):
        # idx +=30
        """Generate and transform data."""
        info = self.prepare_raw_data(idx)
        info_orig = copy.deepcopy(info)
        loaded_data = self.pipeline_load(info)
        orig_img = torch.tensor(loaded_data['img'].copy()[..., [2, 1, 0]]).permute(2, 0, 1)

        # change bounding box
        im_h, im_w = orig_img.shape[-2:]

        if 'K' not in info and 'focal_length' in self.human_data:
            info['K'] = np.eye(3, 3).astype(np.float32)
            raw_focal = self.human_data['focal_length'][idx].reshape(-1)
            info['K'][0, 0] = raw_focal[0]
            info['K'][1, 1] = raw_focal[1]
            info['K'][0, 2] = im_w / 2
            info['K'][1, 2] = im_h / 2
            info['has_K'] = 1


        # loaded_data['center'] = np.array([im_w/2, im_h/2])
        if not self.test_mode:
            loaded_data['center'] = np.array([info['K'][0,2], info['K'][1,2]])
            in_frame_flag = loaded_data['keypoints2d'][:,-1] > 0
            valid_pts = loaded_data['keypoints2d'][in_frame_flag]

            bbox_min_size = min(min(im_h, im_w),
                    2 * np.max([np.abs(valid_pts[:, 0] - im_w / 2).max(), np.abs(valid_pts[:, 1] - im_h / 2).max()]))

            # NOTE: make this a config
            depth_scaling = 0.9 + np.random.rand() * 0.5
            # rand_scaling = 1.
            depth_bbox_size = bbox_min_size * depth_scaling

            pose_scaling = 0.7 + np.random.rand() * 1.3
            pose_bbox_size = bbox_min_size * pose_scaling
        else:
            # loaded_data['center'] = np.array([im_w/2, im_h/2])
            pose_bbox_size = depth_bbox_size = max(im_w, im_h)
        loaded_data_depth = copy.deepcopy(loaded_data)
        loaded_data_pose = copy.deepcopy(loaded_data)

        # input to pose head
        loaded_data_pose['center'] = np.array([im_w/2, im_h/2])
        loaded_data_pose['scale'] = np.array([pose_bbox_size, pose_bbox_size])
        blade_data = self.pipeline_pose(loaded_data_pose)
        aios_img = blade_data['img']
        if not self.test_mode and self.do_black_pad_aug and np.random.rand() < 0.3:
            v_blackout = blade_data['posenet_img_vertical_blackout'] = int(np.random.rand() * 0.2 * aios_img.shape[1])
            h_blackout = blade_data['posenet_img_horizontal_blackout'] = int(np.random.rand() * 0.2 * aios_img.shape[2])
            zero_color = torch.tensor([-2.1179, -2.0357, -1.8044])[:, None, None]
            aios_img[:, -v_blackout:] = aios_img[:, :v_blackout] = zero_color
            aios_img[:, :, -h_blackout:] = aios_img[:, :, :h_blackout] = zero_color
        else:
            if info['dataset_name'] == 'humman':
                pass
            elif info['dataset_name'] == 'pdhuman':
                pass
            elif info['dataset_name'] == 'spec_mtp':
                if aios_img.shape[-2] > 1280:
                    downscale = 1280 / aios_img.shape[-2]
                    aios_img = F.interpolate(aios_img[None], scale_factor=downscale, mode='bilinear', align_corners=False, antialias=True)[0]

        blade_data['img'] = aios_img
        blade_data['posenet_img'] = aios_img
        blade_data['posenet_scale'] = loaded_data_pose['scale']

        loaded_data_depth['center'] = np.array(info_orig['center'])
        loaded_data_depth['scale'] = np.array(info_orig['scale'])
        depth_data = self.pipeline_depth(loaded_data_depth)
        blade_data['depthnet_img'] = depth_data['img']
        blade_data['depthnet_scale'] = loaded_data_depth['scale']
        blade_data['smpl_body_pose'] = depth_data['smpl_body_pose'].view(-1)    # standardize shape
        blade_data['smpl_global_orient'] = depth_data['smpl_global_orient']
        blade_data['smpl_betas'] = depth_data['smpl_betas']
        blade_data['smpl_transl'] = depth_data['smpl_transl']

        if 'K' in info:
            blade_data['orig_K'] = loaded_data_depth['K'].clone()
            if 'is_flipped' not in blade_data:
                blade_data['is_flipped'] = torch.tensor(0)
            flip_flag = -(blade_data['is_flipped'] * 2 - 1)
            blade_data['flip_flag'] = flip_flag.clone()

            K = loaded_data_depth['K']
            K[0, 2] = flip_flag * (K[0, 2] - loaded_data_depth['center'][0])
            K[1, 2] = (K[1, 2] - loaded_data_depth['center'][1])
            scaling = (max(blade_data['img'].shape[-2:]) / loaded_data_depth['scale'].max())
            K[0] *= scaling
            K[1] *= scaling
            K[0, 2] += blade_data['img'].shape[-1] / 2
            K[1, 2] += blade_data['img'].shape[-2] / 2
            blade_data['K'] = K.float()

            K = loaded_data_pose['K'].clone()
            K[0, 2] = flip_flag * (K[0, 2] - loaded_data_pose['center'][0])
            K[1, 2] = (K[1, 2] - loaded_data_pose['center'][1])
            scaling = (max(blade_data['posenet_img'].shape[-2:]) / loaded_data_pose['scale'].max())
            K[0] *= scaling
            K[1] *= scaling
            K[0, 2] += blade_data['posenet_img'].shape[-1] / 2
            K[1, 2] += blade_data['posenet_img'].shape[-2] / 2
            blade_data['posenet_K'] = K.float()

        blade_data['has_SMPLX'] = torch.tensor(info['has_SMPLX'])
        if info['has_SMPLX']:
            # NOTE: store original params, use these to recover correct translation after augmentation
            blade_data['SMPLX_body_pose_orig'] = torch.tensor(info['SMPLX_body_pose']).clone().float()
            blade_data['SMPLX_global_orient_orig'] = torch.tensor(info['SMPLX_global_orient']).clone().float()
            blade_data['SMPLX_right_hand_pose_orig'] = torch.tensor(info['SMPLX_right_hand_pose']).clone().float()
            blade_data['SMPLX_left_hand_pose_orig'] = torch.tensor(info['SMPLX_left_hand_pose']).clone().float()
            blade_data['SMPLX_transl_orig'] = torch.tensor(info['SMPLX_transl']).clone().float()

            # flip params
            if blade_data['is_flipped']:
                info['SMPLX_body_pose'] = _flip_smplx_pose(info['SMPLX_body_pose'].reshape(63)).reshape(21, 3)
                info['SMPLX_global_orient'] = _flip_axis_angle(info['SMPLX_global_orient'].reshape(3)).reshape(1, 3)
                info['SMPLX_right_hand_pose'], info['SMPLX_left_hand_pose'] \
                    = _flip_hand_pose(info['SMPLX_right_hand_pose'], info['SMPLX_left_hand_pose'])

            # rotate params
            info['SMPLX_global_orient'], Rz = _rotate_smpl_pose(info['SMPLX_global_orient'].reshape(-1),
                                                            blade_data['rotation'].item(),
                                                            return_rot_mat=True)
            info['SMPLX_global_orient'] = info['SMPLX_global_orient'].reshape(1, 3)

            # NOTE: translation need to be rotated with pelvis!
            blade_data['SMPLX_rotation_aug'] = torch.tensor(Rz).t().float()
            blade_data['SMPLX_transl'] = torch.tensor(info['SMPLX_transl']).float()
            blade_data['SMPLX_body_pose'] = torch.tensor(info['SMPLX_body_pose']).float()
            blade_data['SMPLX_left_hand_pose'] = torch.tensor(info['SMPLX_left_hand_pose']).float()
            blade_data['SMPLX_right_hand_pose'] = torch.tensor(info['SMPLX_right_hand_pose']).float()
            blade_data['SMPLX_global_orient'] = torch.tensor(info['SMPLX_global_orient']).float()
            blade_data['SMPLX_origin_orient'] = torch.tensor(info['SMPLX_origin_orient']).float()
            blade_data['SMPLX_betas'] = torch.tensor(info['SMPLX_betas']).float()
            blade_data['SMPLX_gender'] = torch.tensor(0)

        blade_data['id_idx'] = idx
        return blade_data

    def evaluate(self,
                 outputs: list,
                 res_folder: str,
                 metric: Optional[Union[str, List[str]]] = 'pa-mpjpe',
                 res = None,
                 **kwargs: dict):
        """Evaluate 3D keypoint results.

        Args:
            outputs (list): results from model inference.
            res_folder (str): path to store results.
            metric (Optional[Union[str, List(str)]]):
                the type of metric. Default: 'pa-mpjpe'
            kwargs (dict): other arguments.
        Returns:
            dict:
                A dict of all evaluation results.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        for metric in metrics:
            if metric not in self.ALLOWED_METRICS:
                raise KeyError(f'metric {metric} is not supported')

        if res is not None:
            pass
        else:
            if res_folder:
                res_file = os.path.join(res_folder, 'results.json')
            # for keeping correctness during multi-gpu test, we sort all results

            res_dict = {}
            entry_dict = dict(
                keypoints='keypoints_3d',
                poses='smpl_pose',
                betas='smpl_beta',
                raw_batch_miou='raw_batch_miou',
                raw_batch_pmiou='raw_batch_pmiou',
                opti_batch_miou='opti_batch_miou',
                opti_batch_pmiou='opti_batch_pmiou',
                opti_batch_miou_w_gt_mask='opti_batch_miou_w_gt_mask',
                opti_batch_pmiou_w_gt_mask='opti_batch_pmiou_w_gt_mask',
                z_error='z_error',
                distortion_error='distortion_error',
                vertices='vertices',
                inv_z_error='inv_z_error',
                xy_error='xy_error',
                f_perc_error='f_perc_error'
            )
            for out in outputs:
                target_id = out['image_idx']
                batch_size = target_id.shape[0]
                for i in range(batch_size):
                    cur_res_dict = {}

                    for k, v in entry_dict.items():
                        if v in out and i in range(len(out[v])):
                            cur_res_dict[k] = out[v][i]

                    res_dict[int(target_id[i])] = cur_res_dict

                    if out.get('estimate_verts', False) and 'vertices' in out and 'pred_keypoints3d' in out:
                        res_dict[int(target_id[i])].update(
                            dict(
                                pred_keypoints3d=out['pred_keypoints3d'][i],
                            ))

            keypoints, poses, betas = [], [], []
            vertices = []
            pred_kp3d = []
            raw_miou = []
            raw_pmiou = []
            opti_miou = []
            opti_pmiou = []
            opti_miou_w_gt_mask = []
            opti_pmiou_w_gt_mask = []
            z_error = []
            distortion_error = []
            inv_z_error = []
            xy_error = []
            f_perc_error = []
            # pred_kp2d = []
            # gt_kp2d = []
            # print(f'res_dict.keys(): {res_dict.keys()}')
            for i in sorted(res_dict.keys()):
                if 'keypoints' in res_dict[i]:
                    keypoints.append(res_dict[i]['keypoints'])
                if 'poses' in res_dict[i]:
                    poses.append(res_dict[i]['poses'])
                if 'betas' in res_dict[i]:
                    betas.append(res_dict[i]['betas'])

                if 'raw_batch_miou' in res_dict[i]:
                    raw_miou.append(res_dict[i]['raw_batch_miou'])
                if 'raw_batch_pmiou' in res_dict[i]:
                    raw_pmiou.append(res_dict[i]['raw_batch_pmiou'])
                if 'opti_batch_miou' in res_dict[i]:
                    opti_miou.append(res_dict[i]['opti_batch_miou'])
                if 'opti_batch_pmiou' in res_dict[i]:
                    opti_pmiou.append(res_dict[i]['opti_batch_pmiou'])
                if 'opti_batch_pmiou_w_gt_mask' in res_dict[i]:
                    opti_miou_w_gt_mask.append(res_dict[i]['opti_batch_miou_w_gt_mask'])
                if 'opti_batch_miou_w_gt_mask' in res_dict[i]:
                    opti_pmiou_w_gt_mask.append(res_dict[i]['opti_batch_pmiou_w_gt_mask'])

                if 'z_error' in res_dict[i]:
                    z_error.append(res_dict[i]['z_error'])
                if 'distortion_error' in res_dict[i]:
                    distortion_error.append(res_dict[i]['distortion_error'])

                if 'inv_z_error' in res_dict[i]:
                    inv_z_error.append(res_dict[i]['inv_z_error'])
                if 'xy_error' in res_dict[i]:
                    xy_error.append(res_dict[i]['xy_error'])
                if 'f_perc_error' in res_dict[i]:
                    f_perc_error.append(res_dict[i]['f_perc_error'])

                # pred_kp2d.append(res_dict[i]['pred_keypoints2d'])
                # gt_kp2d.append(res_dict[i]['gt_keypoints2d'])
                if 'vertices' in res_dict[i]:
                    vertices.append(res_dict[i]['vertices'])
                if 'pred_keypoints3d' in res_dict[i]:
                    pred_kp3d.append(res_dict[i]['pred_keypoints3d'])

            res = dict(
                keypoints=keypoints[:self.num_data],
                poses=poses[:self.num_data],
                betas=betas[:self.num_data],
                raw_miou=raw_miou[:self.num_data],
                raw_pmiou=raw_pmiou[:self.num_data],
                opti_miou=opti_miou[:self.num_data],
                opti_pmiou=opti_pmiou[:self.num_data],
                opti_miou_w_gt_mask=opti_miou_w_gt_mask[:self.num_data],
                opti_pmiou_w_gt_mask=opti_pmiou_w_gt_mask[:self.num_data],
                z_error=z_error[:self.num_data],
                distortion_error=distortion_error[:self.num_data],
                inv_z_error=inv_z_error[:self.num_data],
                xy_error=xy_error[:self.num_data],
                f_perc_error=f_perc_error[:self.num_data],
                #    pred_keypoints2d=pred_kp2d[:self.num_data],
                #    gt_keypoints2d=gt_kp2d[:self.num_data]
            )
            print(f'raw_miou mean: {np.nanmean(raw_miou)}')
            print(f'raw_pmiou mean: {np.nanmean(raw_pmiou)}')
            print(f'opti_miou mean: {np.nanmean(opti_miou)}')
            print(f'opti_pmiou mean: {np.nanmean(opti_pmiou)}')
            print(f'opti_miou_w_gt_mask: {np.nanmean(opti_miou_w_gt_mask)}')
            print(f'opti_pmiou_w_gt_mask: {np.nanmean(opti_pmiou_w_gt_mask)}')
            print(f'z_error: {np.nanmean(z_error)}')
            print(f'z stddev: {np.nanstd(z_error)}')

            print(f'inv_z_error: {np.nanmean(inv_z_error)}')
            print(f'xy_error: {np.nanmean(xy_error)}')
            print(f'f_perc_error: {np.nanmean(f_perc_error)}')

            if len(vertices):
                res['esitmated_vertices'] = vertices[:self.num_data]
            if len(pred_kp3d):
                res['estimated_keypoints3d'] = pred_kp3d[:self.num_data]
            if res_folder:
                mmcv.dump(res, res_file)

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                _nv_tuples, mpjpe_array = self._report_mpjpe(res)
            elif _metric == 'pa-mpjpe':
                _nv_tuples, pampjpe_array = self._report_mpjpe(res, metric='pa-mpjpe')
            elif _metric == '3dpck':
                _nv_tuples = self._report_3d_pck(res)
            elif _metric == 'pa-3dpck':
                _nv_tuples = self._report_3d_pck(res, metric='pa-3dpck')
            elif _metric == '3dauc':
                _nv_tuples = self._report_3d_auc(res)
            elif _metric == 'pa-3dauc':
                _nv_tuples = self._report_3d_auc(res, metric='pa-3dauc')
            elif _metric == 'pve':
                _nv_tuples, pve_array = self._report_pve(res)
            elif _metric == 'pa-pve':
                _nv_tuples, papve_array = self._report_pve(res, metric='pa-pve')
            elif _metric == 'ihmr':
                _nv_tuples = self._report_ihmr(res)

            elif _metric == 'raw_miou':
                _nv_tuples = self._report_miou(res, metric='raw_miou')
            elif _metric == 'raw_pmiou':
                _nv_tuples = self._report_miou(res, metric='raw_pmiou')
            elif _metric == 'opti_miou':
                _nv_tuples = self._report_miou(res, metric='opti_miou')
            elif _metric == 'opti_pmiou':
                _nv_tuples = self._report_miou(res, metric='opti_pmiou')
            elif _metric == 'opti_miou_w_gt_mask':
                _nv_tuples = self._report_miou(res, metric='opti_miou_w_gt_mask')
            elif _metric == 'opti_pmiou_w_gt_mask':
                _nv_tuples = self._report_miou(res, metric='opti_pmiou_w_gt_mask')

            elif _metric == 'z_error':
                _nv_tuples = self._report_z_error(res, metric='z_error')
            elif _metric == 'distortion_error':
                _nv_tuples = self._report_distortion_error(res, metric='distortion_error')

            elif _metric == 'inv_z_error':
                _nv_tuples = self._report_key_error(res, metric='inv_z_error')
            elif _metric == 'xy_error':
                _nv_tuples = self._report_key_error(res, metric='xy_error')
            elif _metric == 'f_perc_error':
                _nv_tuples = self._report_key_error(res, metric='f_perc_error')
            else:
                raise NotImplementedError
            name_value_tuples.extend(_nv_tuples)

        # print(f'xy_error all: {xy_error}')
        # print(f'PA-MPJPE error all: {pampjpe_array}')
        # print(f'MPJPE error all: {mpjpe_array}')
        # print(f'PVE error all: {pve_array}')

        # error_array_fn = f'error_data_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}.npz'
        # print(f'Saving res & error arrays to {error_array_fn}')
        # np.savez(error_array_fn,
        #          pampjpe_error=pampjpe_array,
        #          mpjpe_error=mpjpe_array,
        #          pve_error=pve_array,
        #          res=res)


        name_value_tuples.extend([('total samples', len(res['z_error']))])
        name_value_tuples.extend([('n_detection_failed', len(res['z_error']) - len(res['raw_miou']))])
        # opti_miou_w_gt_mask = []
        # opti_pmiou_w_gt_mask = []
        # z_error = []

        name_value = OrderedDict(name_value_tuples)
        return name_value, res

    def _report_miou(self, res_file, metric='miou', body_part=''):
        """Cauculate mean per joint position error (MPJPE) or its variants PA-
        MPJPE.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (PA-MPJPE)
        """
        miou = res_file[metric]
        error = np.nanmean(np.array(miou))
        err_name = metric.upper()
        if body_part != '':
            err_name = body_part.upper() + ' ' + err_name

        info_str = [(err_name, error)]

        return info_str

    def _report_z_error(self, res_file, metric='z_error', body_part=''):
        """Cauculate mean per joint position error (MPJPE) or its variants PA-
        MPJPE.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (PA-MPJPE)
        """
        abs_z_error = np.abs(np.array(res_file[metric]))
        error = np.nanmean(abs_z_error)
        error_std = np.nanstd(abs_z_error)
        err_name = metric.upper()
        if body_part != '':
            err_name = body_part.upper() + ' ' + err_name

        info_str = [(err_name, error), (err_name + '_STDDEV', error_std)]

        return info_str

    def _report_key_error(self, res_file, metric='z_error'):
        """Cauculate mean per joint position error (MPJPE) or its variants PA-
        MPJPE.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (PA-MPJPE)
        """
        z_error = res_file[metric]
        error = np.nanmean(np.abs(np.array(z_error)))
        err_name = metric.upper()

        info_str = [(err_name, error)]

        return info_str

    def _report_distortion_error(self, res_file, metric='distortion_error', body_part=''):
        """Cauculate mean per joint position error (MPJPE) or its variants PA-
        MPJPE.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (PA-MPJPE)
        """
        distortion_error = res_file[metric]
        error = np.abs(np.array(distortion_error)).mean()
        error_std = np.abs(np.array(distortion_error)).std()
        err_name = metric.upper()
        if body_part != '':
            err_name = body_part.upper() + ' ' + err_name

        info_str = [(err_name, error), (err_name+'_STDDEV', error_std)]

        return info_str

    def _parse_result(self, res, mode='keypoint', body_part=None):
        """Parse results."""

        if mode == 'vertice':
            # gt
            gt_beta, gt_pose, gt_global_orient, gender = [], [], [], []
            gt_smpl_dict = self.human_data['smpl']
            for idx in range(self.num_data):
                gt_beta.append(gt_smpl_dict['betas'][idx])
                gt_pose.append(gt_smpl_dict['body_pose'][idx])
                gt_global_orient.append(gt_smpl_dict['global_orient'][idx])
                if 'meta' in self.human_data:
                    if self.human_data['meta']['gender'][idx] == 'm':
                        gender.append(0)
                    else:
                        gender.append(1)
                else:
                    gender.append(-1)
            gt_beta = torch.FloatTensor(gt_beta)
            gt_pose = torch.FloatTensor(gt_pose).view(-1, 69)
            gt_global_orient = torch.FloatTensor(gt_global_orient)
            gender = torch.Tensor(gender)
            gt_output = self.body_model(betas=gt_beta,
                                        body_pose=gt_pose,
                                        global_orient=gt_global_orient,
                                        gender=gender)
            gt_vertices = gt_output['vertices'].detach().cpu().numpy() * 1000.
            gt_mask = np.ones(gt_vertices.shape[:-1])
            # pred
            pred_pose = torch.FloatTensor(res['poses'])
            pred_beta = torch.FloatTensor(res['betas'])

            if 'esitmated_vertices' in res:
                right_hip_idx = get_keypoint_idx('right_hip_extra', 'h36m')
                left_hip_idx = get_keypoint_idx('left_hip_extra', 'h36m')

                gt_keypoints3d = gt_output['joints']
                gt_pelvis = (gt_keypoints3d[:, left_hip_idx] +
                             gt_keypoints3d[:, right_hip_idx]) / 2
                gt_vertices = (gt_output['vertices'] - gt_pelvis.view(-1, 1, 3)).detach().cpu().numpy() * 1000.

                pred_keypoints3d = np.array(res['keypoints'])
                pred_pelvis = (pred_keypoints3d[:, left_hip_idx] + pred_keypoints3d[:, right_hip_idx]) / 2
                pred_vertices = (np.stack(res['esitmated_vertices'], 0) - pred_pelvis[:, None]) * 1000.

            else:
                assert False, "not implemented"
                pred_output = self.body_model(
                    betas=pred_beta,
                    body_pose=pred_pose[:, 1:],
                    global_orient=pred_pose[:, 0].unsqueeze(1),
                    pose2rot=False,
                    gender=None)
                pred_vertices = (pred_output['vertices'] - pred_output['joints'][:,:1]).detach().cpu().numpy() * 1000.

            assert len(pred_vertices) == self.num_data, f"{len(pred_keypoints3d)} (pred_keypoints3d)!= {self.num_data} (self.num_data)"

            return pred_vertices, gt_vertices, gt_mask

        elif mode == 'keypoint':
            if 'estimated_keypoints3d' in res:
                pred_keypoints3d = res['estimated_keypoints3d']
            else:
                pred_keypoints3d = res['keypoints']

            assert len(pred_keypoints3d) == self.num_data, f"{len(pred_keypoints3d)} (pred_keypoints3d)!= {self.num_data} (self.num_data)"

            # (B, 17, 3)
            pred_keypoints3d = np.array(pred_keypoints3d)

            if self.dataset_name in [
                    'pdhuman', 'pw3d', 'lspet', 'humman', 'spec_mtp', 'h36m'
            ]:
                # print('testing h36m')
                betas = []
                body_pose = []
                global_orient = []
                gender = []
                smpl_dict = self.human_data['smpl']
                for idx in range(self.num_data):
                    betas.append(smpl_dict['betas'][idx])
                    body_pose.append(smpl_dict['body_pose'][idx])
                    global_orient.append(smpl_dict['global_orient'][idx])
                    if 'meta' in self.human_data:
                        if self.human_data['meta']['gender'][idx] == 'm':
                            gender.append(0)
                        else:
                            gender.append(1)
                    else:
                        gender.append(-1)
                betas = torch.FloatTensor(betas)
                body_pose = torch.FloatTensor(body_pose).view(-1, 69)
                global_orient = torch.FloatTensor(global_orient)
                gender = torch.Tensor(gender)
                gt_output = self.body_model(betas=betas,
                                            body_pose=body_pose,
                                            global_orient=global_orient,
                                            gender=gender)
                gt_keypoints3d = gt_output['joints'].detach().cpu().numpy()
                gt_keypoints3d_mask = np.ones((len(pred_keypoints3d), 24))
            elif self.dataset_name in ['h36m']:
                gt_keypoints3d = self.human_data[
                    'keypoints3d'][:self.num_data, :, :3]
                gt_keypoints3d_mask = np.ones(
                    (len(pred_keypoints3d), pred_keypoints3d.shape[-2]))

            else:
                raise NotImplementedError()

            # SMPL_49 only!
            if gt_keypoints3d.shape[1] == 49:
                assert pred_keypoints3d.shape[1] == 49

                gt_keypoints3d = gt_keypoints3d[:, 25:, :]
                pred_keypoints3d = pred_keypoints3d[:, 25:, :]

                joint_mapper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
                gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
                pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]

                # we only evaluate on 14 lsp joints
                pred_pelvis = (pred_keypoints3d[:, 2] +
                               pred_keypoints3d[:, 3]) / 2
                gt_pelvis = (gt_keypoints3d[:, 2] + gt_keypoints3d[:, 3]) / 2

            # H36M for testing!
            elif gt_keypoints3d.shape[1] == 17:
                assert pred_keypoints3d.shape[1] == 17

                H36M_TO_J17 = [
                    6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9
                ]
                H36M_TO_J14 = H36M_TO_J17[:14]
                joint_mapper = H36M_TO_J14

                pred_pelvis = pred_keypoints3d[:, 0]
                gt_pelvis = gt_keypoints3d[:, 0]

                gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
                pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]

            # keypoint 24
            elif gt_keypoints3d.shape[1] == 24:
                assert pred_keypoints3d.shape[1] == 24

                joint_mapper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
                gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
                pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]

                # we only evaluate on 14 lsp joints
                pred_pelvis = (pred_keypoints3d[:, 2] +
                               pred_keypoints3d[:, 3]) / 2
                gt_pelvis = (gt_keypoints3d[:, 2] + gt_keypoints3d[:, 3]) / 2

            else:
                pass
            if not 'estimated_keypoints3d' in res:
                pred_keypoints3d = (pred_keypoints3d - pred_pelvis[:, None, :]) * 1000
            else:
                pred_keypoints3d = pred_keypoints3d * 1000
            gt_keypoints3d = (gt_keypoints3d - gt_pelvis[:, None, :]) * 1000

            gt_keypoints3d_mask = gt_keypoints3d_mask[:, joint_mapper] > 0

            return pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask

    def _parse_result2d(self, res):
        """Parse results."""
        pred_keypoints2d = np.array(res['pred_keypoints2d'])
        gt_keypoints2d = np.array(res['gt_keypoints2d'])
        gt_keypoints2d_mask = gt_keypoints2d[..., 2:3]
        return pred_keypoints2d, gt_keypoints2d, gt_keypoints2d_mask
