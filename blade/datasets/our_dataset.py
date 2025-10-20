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


import mmcv, os, torch, shutil, re
from pathlib import Path
import numpy as np
from glob import glob
from torchvision import transforms
from typing import Optional, Union, List
from collections import OrderedDict
from blade.datasets.human_image_dataset import HumanImageDataset
from mmcv.runner import get_dist_info
from blade.configs.base import root
from blade.utils.helpers import get_global_rank, get_local_rank
from blade.models.body_models.mappings import get_keypoint_idx
from blade.utils.bbox_utils import kp2d_to_bbox
from blade.datasets.pipelines.compose import Compose
import torch.distributed as dist

_FIN_RE = re.compile(r'^(seq_\d{6}_)(FinalImage_)?(\d{4}\.png)$')



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


class OurDataset(HumanImageDataset):
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
    ):

        assert cache_data_path is not None, "give cache_data_path to dataset loader"

        self.data_prefix = data_prefix
        self.dataset_name = dataset_name

        ann_prefix = os.path.join(self.data_prefix, 'preprocessed_datasets')
        self.ann_file = os.path.join(ann_prefix, ann_file)
        self.ann_data = torch.load(self.ann_file)

        super().__init__(data_prefix, pipeline, dataset_name, body_model,
                         ann_file, convention, cache_data_path, test_mode)

        self.start_index = start_index
        if start_index is not None:
            self.num_data = self.num_data - start_index
        if num_data is not None:
            self.num_data = num_data
        self.is_distorted = is_distorted
        self.pipeline_load = Compose(pipeline[:1])
        self.pipeline_rest = Compose(pipeline[1:])
        self.dinov2_transform = make_depth_transform()

        world_size = get_dist_info()[1]
        global_rank = get_global_rank()
        local_rank = get_local_rank()
        if global_rank == 0 and local_rank == 0:
            if not os.path.exists(os.path.join(root, 'cache/our_dataset.npz')):
                shutil.copyfile(os.path.join(root, 'mmhuman_data/preprocessed_datasets/our_dataset.npz'),
                                os.path.join(root, 'cache/our_dataset.npz'))
        if world_size > 1:
            dist.barrier()

        self.num_data = len(self.ann_data['img_fn'])

    def load_annotations(self):
        pass

    def prepare_raw_data(self, idx: int):
        """Get item from self.human_data."""
        # self.human_data = self.ann_data[idx]
        info = {}
        info['img_prefix'] = None
        info['sample_idx'] = idx
        info['dataset_name'] = self.dataset_name
        info['image_path'] = os.path.join(self.data_prefix, 'datasets', self.dataset_name, 'png',
                                          self.ann_data['sequence_name'][idx], self.ann_data['img_fn'][idx])

        # check
        # p = Path(info['image_path'])
        # if p.exists():
        #     # print(f"image exists: {info['image_path']}")
        #     pass
        # else:
        #     # print(f"image doesn't exist: {info['image_path']}")
        #     name = p.name
        #     m = _FIN_RE.match(name)
        #     if not m:
        #         # Name doesn't match expected pattern; nothing to toggle
        #         assert False, f"could not parse {info['image_path']}"
        #     prefix, has_final, tail = m.group(1), m.group(2), m.group(3)
        #     if has_final:
        #         # Try without FinalImage_
        #         info['image_path'] = p.with_name(prefix + tail)
        #     else:
        #         # Try with FinalImage_
        #         info['image_path'] = p.with_name(prefix + 'FinalImage_' + tail)
        # info['image_path'] = str(info['image_path'])

        info['has_pelvis_camcoord'] = 1
        info['pelvis_camcoord'] = self.ann_data['pelvis_camcoord'][idx][...,0]

        f = self.ann_data['focal_length'][idx]
        info['ori_focal_length'] = f
        info['has_focal_length'] = 1

        h, w = self.ann_data['hw'][idx]
        info['bbox_xywh'] = [0, 0, w, h, 1.]


        info['K'] = torch.tensor([[f, 0, w/2],[0, f, h/2],[0, 0, 1]]).float()
        info['has_K'] = 1

        info['has_smpl'] = 0
        smpl_dict = {}

        # in later modules, we will check validity of each keypoint by
        # its confidence. Therefore, we do not need the mask of keypoints.

        # SMPL & keypoints
        info['keypoints2d'] = np.zeros((self.num_keypoints, 3))
        info['has_keypoints2d'] = 0
        info['keypoints3d'] = np.zeros((self.num_keypoints, 4))
        info['has_keypoints3d'] = 0

        if 'bbox_xywh' in info:
            x, y, w, h, _ = info['bbox_xywh']
            cx = x + w / 2
            cy = y + h / 2
            w = h = max(w, h)
            info['center'] = np.array([cx, cy])
            info['scale'] = np.array([w, h])
        elif 'keypoints2d' in self.ann_data:
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

        info['smpl_body_pose'] = np.concatenate([self.ann_data['body_pose'][idx],
                                            self.ann_data['left_hand_pose'][idx][:,:3], self.ann_data['right_hand_pose'][idx][:,:3]],1).reshape(23,3)
        info['smpl_global_orient'] = self.ann_data['global_orient'][idx].reshape(3)
        info['smpl_origin_orient'] = info['smpl_global_orient'].copy().reshape(3)
        info['smpl_betas'] = self.ann_data['betas'][idx].reshape(10)

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
        info = self.prepare_raw_data(idx)
        loaded_data = self.pipeline_load(info)
        orig_img = torch.tensor(loaded_data['img'].copy()[..., [2, 1, 0]]).permute(2, 0, 1)
        blade_data = self.pipeline_rest(loaded_data)
        blade_data['pelvis_camcoord'] = torch.tensor(info['pelvis_camcoord'])
        blade_data['has_pelvis_camcoord'] = torch.tensor(info['has_pelvis_camcoord'])
        blade_data['depthnet_img'] = blade_data['img']
        # blade_data['orig_img'] = orig_img
        # blade_data['id_idx'] = idx
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
        for i in range(self.num_data):
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
        )
        if len(vertices):
            res['esitmated_vertices'] = vertices[:self.num_data]
        if len(pred_kp3d):
            res['estimated_keypoints3d'] = pred_kp3d[:self.num_data]

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
