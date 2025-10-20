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

import mmcv, os, torch, shutil, copy, gzip, pickle, tempfile, socket
import numpy as np
from glob import glob
from torchvision import transforms
from typing import Optional, Union, List
from collections import OrderedDict
from blade.datasets.human_image_dataset import HumanImageDataset
from mmcv.runner import get_dist_info
from blade.configs.base import root
from blade.utils.helpers import get_global_rank, get_local_rank
from blade.datasets.pipelines.transforms import _rotate_smpl_pose, _flip_smplx_pose, _flip_axis_angle, _flip_hand_pose

from blade.models.body_models.mappings import get_keypoint_idx
from blade.utils.bbox_utils import kp2d_to_bbox
from blade.datasets.utils import pad_to_square
from blade.datasets.pipelines.compose import Compose
import torch.distributed as dist

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


class OurDataset_SMPLX(HumanImageDataset):
    ALLOWED_METRICS = {
        'mpjpe', 'pa-mpjpe', 'pve', '3dpck', 'pa-3dpck', '3dauc', 'pa-3dauc',
        'ihmr', 'pa-pve', 'miou', 'pmiou', 'z_error'
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
        do_black_pad_aug = False,
        is_test = False,
        data_folder = 'our_dataset_v6'
    ):

        assert cache_data_path is not None, "give cache_data_path to dataset loader"

        self.data_prefix = data_prefix
        self.dataset_name = dataset_name

        ann_prefix = os.path.join(self.data_prefix, 'preprocessed_datasets')
        self.ann_file = os.path.join(ann_prefix, ann_file)
        self.is_test = is_test
        if self.is_test:
            self.ann_data = torch.load(self.ann_file)
        else:
            print(f"loading BEDLAMCC from {self.ann_file}")
            # ann_data = np.load(self.ann_file, allow_pickle=True)
            ann_data = torch.load(self.ann_file)
            self.ann_data = {}
            for k, v in ann_data.items():
                print(f"key is {k}")
                if k in ['img_fn', 'gender']:
                    self.ann_data[k] = list(v)
                elif k == 'center':
                    self.ann_data[k] = torch.tensor(v)
                elif k == 'smplx':
                    self.ann_data[k] = {}
                    # smplx_dict = v.item()
                    for smplx_k, smplx_v in v.items():
                        print(f"key is {k} - {smplx_k}")
                        self.ann_data[k][smplx_k] = torch.tensor(smplx_v)
                else:
                    self.ann_data[k] = torch.tensor(v)
        # self.ann_data = torch.load(self.ann_file)


        # self.ann_data = torch.load(self.ann_file)
        # with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as tmp_file:
        #     tmp_file_path = tmp_file.name
        #     shutil.copy(self.ann_file, tmp_file_path)
        #     self.ann_data = torch.load(tmp_file_path, allow_pickle=True)
        # os.remove(tmp_file_path)

        # print(f"super init")
        super().__init__(data_prefix, pipeline, dataset_name, body_model,
                         ann_file, convention, cache_data_path, test_mode)
        # print(f"pipeline init")
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
        world_size = get_dist_info()[1]
        global_rank = get_global_rank()
        local_rank = get_local_rank()

        self.num_data = len(self.ann_data['img_fn'])
        print(f"num_data = {self.num_data}")
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        print(f"copying to cache/")
        if global_rank == 0 and local_rank == 0:
            print(f"rank 0 copying to cache/")
            shutil.copyfile(os.path.join(root, 'mmhuman_data', 'preprocessed_datasets', ann_file),
                            os.path.join(root, 'cache', ann_file))

        if world_size > 1:
            print(f"barrier")
            dist.barrier()

        print(f"loading data into memory")
        self.image_path = []
        for im_fn in self.ann_data['img_fn']:
            self.image_path.append(os.path.join(root, 'mmhuman_data', 'datasets', data_folder,'png',
                                                im_fn.split('/')[-2], im_fn.split('/')[-1]))


        self.center = self.ann_data['center']
        self.scale = torch.tensor(self.ann_data['scale']).numpy()
        self.focal_length = torch.tensor(self.ann_data['focal_length']).numpy()

        # self.smplx_dict = self.ann_data['smplx'].item()
        self.SMPLX_betas = self.ann_data['smplx']['betas'].numpy()
        self.SMPLX_body_pose = self.ann_data['smplx']['body_pose'].numpy()
        self.SMPLX_global_orient = self.ann_data['smplx']['global_orient'].numpy()
        self.SMPLX_origin_orient = self.ann_data['smplx']['global_orient'].numpy()
        self.SMPLX_left_hand_pose = self.ann_data['smplx']['left_hand_pose'].numpy()
        self.SMPLX_right_hand_pose = self.ann_data['smplx']['right_hand_pose'].numpy()
        self.SMPLX_transl = self.ann_data['smplx']['transl'].numpy()
        self.SMPLX_gender = self.ann_data['gender']

        print(f'total number of samples in our dataset: {self.num_data}')


    def load_annotations(self):
        pass

    def prepare_raw_data(self, idx: int):
        """Get item from self.human_data."""
        # self.human_data = self.ann_data[idx]
        # print(f"load sample {idx}")
        info = {}
        info['img_prefix'] = None
        info['sample_idx'] = idx
        info['dataset_name'] = self.dataset_name
        info['image_path'] = self.image_path[idx]
        info['center'] = self.center[idx]
        info['scale'] = self.scale[idx]
        info['has_SMPLX'] = 1
        info['SMPLX_betas'] = self.SMPLX_betas[idx].reshape(-1)
        info['SMPLX_body_pose'] = self.SMPLX_body_pose[idx].reshape(-1,3)
        info['SMPLX_global_orient'] = self.SMPLX_global_orient[idx].reshape(-1)
        info['SMPLX_origin_orient'] = self.SMPLX_origin_orient[idx].reshape(-1)
        info['SMPLX_left_hand_pose'] = self.SMPLX_left_hand_pose[idx].reshape(-1,3)
        info['SMPLX_right_hand_pose'] = self.SMPLX_right_hand_pose[idx].reshape(-1,3)
        info['SMPLX_transl'] = self.SMPLX_transl[idx].reshape(-1)
        if self.SMPLX_gender[idx] == 'male':
            info['SMPLX_gender'] = 1
        elif self.SMPLX_gender[idx] == 'female':
            info['SMPLX_gender'] = -1
        elif self.SMPLX_gender[idx] == 'neutral':
            info['SMPLX_gender'] = 0
        f = self.focal_length[idx]
        info['ori_focal_length'] = f.reshape(1)
        info['has_focal_length'] = 1
        info['has_K'] = 0

        info['bbox_xywh'] = [0, 0, 0, 0, 1.]

        info['keypoints2d'] = np.zeros((self.num_keypoints, 3))
        info['has_keypoints2d'] = 0
        info['keypoints3d'] = np.zeros((self.num_keypoints, 4))
        info['has_keypoints3d'] = 0

        info['center'] = self.center[idx]
        info['scale'] = np.array([self.scale[idx], self.scale[idx]]) * 1.25 # expand bounding box


        info['has_smpl'] = 0
        smpl_dict = {}
        if 'body_pose' in smpl_dict:
            info['smpl_body_pose'] = smpl_dict['body_pose']
        else:
            info['smpl_body_pose'] = np.zeros((23 * 3))
        if 'global_orient' in smpl_dict:
            info['smpl_global_orient'] = smpl_dict['global_orient']
        else:
            info['smpl_global_orient'] = np.zeros((3))
        info['smpl_origin_orient'] = info['smpl_global_orient'].copy()
        if 'betas' in smpl_dict:
            info['smpl_betas'] = smpl_dict['betas']
        else:
            info['smpl_betas'] = np.zeros((10))
        if 'betas_neutral' in smpl_dict:
            if not self.test_mode:  #for pw3d training
                info['smpl_betas'] = smpl_dict['betas_neutral']
        if 'transl' in smpl_dict:
            info['smpl_transl'] = smpl_dict['transl'].astype(np.float32)
            info['has_transl'] = 1
        else:
            info['smpl_transl'] = np.zeros((3)).astype(np.float32)
            info['has_transl'] = 0


        info['distortion_max'] = 1.0
        info['is_distorted'] = 0

        return info

    def prepare_data(self, idx: int):
        """Generate and transform data."""
        # print(f"prepare_data {idx}")
        info = self.prepare_raw_data(idx)
        info_orig = copy.deepcopy(info)
        loaded_data = self.pipeline_load(info)
        orig_img = torch.tensor(loaded_data['img'].copy()[..., [2, 1, 0]]).permute(2, 0, 1)

        # change bounding box
        im_h, im_w = orig_img.shape[-2:]

        if 'K' not in info and 'ori_focal_length' in info:
            info['K'] = np.eye(3, 3).astype(np.float32)
            raw_focal = info['ori_focal_length']
            info['K'][0, 0] = raw_focal
            info['K'][1, 1] = raw_focal
            info['K'][0, 2] = im_w / 2
            info['K'][1, 2] = im_h / 2
            info['has_K'] = 1

        # print(f"start pipline {idx}")
        # posenet center crop
        if not self.test_mode:
            cx, cy = info['K'][0,2], info['K'][1,2]
            loaded_data['center'] = np.array([cx, cy])
            orig_bbox_scale = loaded_data['scale']

            max_x = np.abs(2*((loaded_data['center'][0] - cx)) + orig_bbox_scale[0] / 2)
            max_y = np.abs(2*((loaded_data['center'][1] - cy)) + orig_bbox_scale[1] / 2)


            bbox_min_size = min(max(im_h, im_w), max(max_x, max_y))

            # NOTE: make this a config
            depth_scaling = 0.9 + np.random.rand() * 0.5
            # rand_scaling = 1.
            depth_bbox_size = bbox_min_size * depth_scaling

            pose_scaling = 1.2 + np.random.rand() * 0.4
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

        blade_data['img'] = aios_img
        blade_data['posenet_img'] = aios_img
        blade_data['posenet_scale'] = loaded_data_pose['scale']

        loaded_data_depth['center'] = info_orig['center'].numpy()
        loaded_data_depth['scale'] = info_orig['scale']
        depth_data = self.pipeline_depth(loaded_data_depth)
        blade_data['depthnet_img'] = depth_data['img']
        blade_data['depthnet_scale'] = loaded_data_depth['scale']
        blade_data['smpl_body_pose'] = depth_data['smpl_body_pose'].view(-1)    # standardize shape
        blade_data['smpl_global_orient'] = depth_data['smpl_global_orient']
        # blade_data['smpl_origin_orient'] = depth_data['smpl_origin_orient']
        blade_data['smpl_betas'] = depth_data['smpl_betas']
        blade_data['smpl_transl'] = depth_data['smpl_transl']

        # print(f"dataaug {idx}")
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
            blade_data['SMPLX_global_orient_orig'] = torch.tensor(info['SMPLX_global_orient']).clone().float()[None]
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
            info['SMPLX_global_orient'] = np.array(info['SMPLX_global_orient'])
            info['SMPLX_global_orient'], Rz = _rotate_smpl_pose(info['SMPLX_global_orient'].reshape(-1),
                                                            blade_data['rotation'].item(),
                                                            return_rot_mat=True)
            info['SMPLX_global_orient'] = info['SMPLX_global_orient'].reshape(1, 3)

            blade_data['SMPLX_rotation_aug'] = torch.tensor(Rz).t().float()

            blade_data['SMPLX_transl'] = torch.tensor(info['SMPLX_transl']).float()
            blade_data['SMPLX_body_pose'] = torch.tensor(info['SMPLX_body_pose']).float()
            blade_data['SMPLX_left_hand_pose'] = torch.tensor(info['SMPLX_left_hand_pose']).float()
            blade_data['SMPLX_right_hand_pose'] = torch.tensor(info['SMPLX_right_hand_pose']).float()
            blade_data['SMPLX_global_orient'] = torch.tensor(info['SMPLX_global_orient']).float()
            blade_data['SMPLX_origin_orient'] = torch.tensor(info['SMPLX_origin_orient']).float()
            blade_data['SMPLX_betas'] = torch.tensor(info['SMPLX_betas']).float()
            blade_data['SMPLX_gender'] = torch.tensor(info['SMPLX_gender'])

        blade_data['id_idx'] = idx
        return blade_data

    def evaluate(self,
                 outputs: list,
                 res_folder: str,
                 metric: Optional[Union[str, List[str]]] = 'pa-mpjpe',
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
        if res_folder:
            res_file = os.path.join(res_folder, 'result_keypoints.json')
        # for keeping correctness during multi-gpu test, we sort all results

        res_dict = {}
        entry_dict = dict(
            keypoints='keypoints_3d',
            poses='smpl_pose',
            betas='smpl_beta',
            batch_miou='batch_miou',
            batch_pmiou='batch_pmiou',
            z_error='z_error',
            mpjpe='mpjpe',
            pve='pve'
        )
        for out in outputs:
            target_id = out['image_idx']
            batch_size = target_id.shape[0]
            for i in range(batch_size):
                cur_res_dict = {}

                for k, v in entry_dict.items():
                    if v in out:
                        cur_res_dict[k] = out[v][i]

                res_dict[int(target_id[i])] = cur_res_dict

                if out.get('estimate_verts', False) and 'vertices' in out and 'pred_keypoints3d' in out:
                    res_dict[int(target_id[i])].update(
                        dict(
                            vertices=out['vertices'][i],
                            pred_keypoints3d=out['pred_keypoints3d'][i],
                        ))

        keypoints, poses, betas = [], [], []
        vertices = []
        pred_kp3d = []
        miou = []
        pmiou = []
        z_error = []
        mpjpe, pve = [], []
        # pred_kp2d = []
        # gt_kp2d = []
        for i in range(self.num_data):
            if 'mpjpe' in res_dict[i]:
                mpjpe.append(res_dict[i]['mpjpe'])
            if 'pve' in res_dict[i]:
                pve.append(res_dict[i]['pve'])
            if 'keypoints' in res_dict[i]:
                keypoints.append(res_dict[i]['keypoints'])
            if 'poses' in res_dict[i]:
                poses.append(res_dict[i]['poses'])
            if 'betas' in res_dict[i]:
                betas.append(res_dict[i]['betas'])
            if 'batch_miou' in res_dict[i]:
                miou.append(res_dict[i]['batch_miou'])
            if 'batch_pmiou' in res_dict[i]:
                pmiou.append(res_dict[i]['batch_pmiou'])
            if 'z_error' in res_dict[i]:
                z_error.append(res_dict[i]['z_error'])
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
            miou=miou[:self.num_data],
            pmiou=pmiou[:self.num_data],
            z_error=z_error[:self.num_data]
            #    pred_keypoints2d=pred_kp2d[:self.num_data],
            #    gt_keypoints2d=gt_kp2d[:self.num_data]
        )
        if len(vertices):
            res['esitmated_vertices'] = vertices[:self.num_data]
        if len(pred_kp3d):
            res['estimated_keypoints3d'] = pred_kp3d[:self.num_data]
        # if res_folder:
        #     mmcv.dump(res, res_file)

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                _nv_tuples = self._report_mpjpe(res)
            elif _metric == 'pa-mpjpe':
                _nv_tuples = self._report_mpjpe(res, metric='pa-mpjpe')
            elif _metric == '3dpck':
                _nv_tuples = self._report_3d_pck(res)
            elif _metric == 'pa-3dpck':
                _nv_tuples = self._report_3d_pck(res, metric='pa-3dpck')
            elif _metric == '3dauc':
                _nv_tuples = self._report_3d_auc(res)
            elif _metric == 'pa-3dauc':
                _nv_tuples = self._report_3d_auc(res, metric='pa-3dauc')
            elif _metric == 'pve':
                _nv_tuples = self._report_pve(res, metric='pa-pve')
            elif _metric == 'pa-pve':
                _nv_tuples = self._report_pve(res)
            elif _metric == 'ihmr':
                _nv_tuples = self._report_ihmr(res)
            elif _metric == 'miou':
                _nv_tuples = self._report_miou(res)
            elif _metric == 'pmiou':
                _nv_tuples = self._report_miou(res, metric='pmiou')
            elif _metric == 'z_error':
                _nv_tuples = self._report_z_error(res, metric='z_error')
            else:
                raise NotImplementedError
            name_value_tuples.extend(_nv_tuples)

        name_value = OrderedDict(name_value_tuples)
        return name_value

    def _report_miou(self, res_file, metric='miou', body_part=''):
        """Cauculate mean per joint position error (MPJPE) or its variants PA-
        MPJPE.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (PA-MPJPE)
        """
        miou = res_file[metric]
        error = np.array(miou).mean()
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
        z_error = res_file[metric]
        error = np.abs(np.array(z_error)).mean()
        error_std = np.abs(np.array(z_error)).std()
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
                pred_vertices = np.stack(res['esitmated_vertices'], 0) * 1000.

                gt_keypoints3d = gt_output['joints']

                right_hip_idx = get_keypoint_idx('right_hip_extra', 'h36m')
                left_hip_idx = get_keypoint_idx('left_hip_extra', 'h36m')

                gt_pelvis = (gt_keypoints3d[:, left_hip_idx] +
                             gt_keypoints3d[:, right_hip_idx]) / 2
                gt_vertices = gt_vertices - gt_pelvis.view(
                    -1, 1, 3).detach().cpu().numpy() * 1000.

            else:
                pred_output = self.body_model(
                    betas=pred_beta,
                    body_pose=pred_pose[:, 1:],
                    global_orient=pred_pose[:, 0].unsqueeze(1),
                    pose2rot=False,
                    gender=None)
                pred_vertices = pred_output['vertices'].detach().cpu().numpy(
                ) * 1000.

            assert len(pred_vertices) == self.num_data

            return pred_vertices, gt_vertices, gt_mask

        elif mode == 'keypoint':
            if 'estimated_keypoints3d' in res:
                pred_keypoints3d = res['estimated_keypoints3d']
            else:
                pred_keypoints3d = res['keypoints']

            assert len(pred_keypoints3d) == self.num_data

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
                pred_keypoints3d = (pred_keypoints3d -
                                    pred_pelvis[:, None, :]) * 1000
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