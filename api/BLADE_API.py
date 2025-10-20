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
import argparse, os, mmcv, torch
import os.path as osp

# from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint

# from mmhuman3d.apis import multi_gpu_test, single_gpu_test
from blade.datasets.builder import build_dataloader, build_dataset
from blade.models.architectures.builder import build_architecture
from blade.configs.base import inference_pipeline_batchof1


def single_gpu_test(model, data_loader):
    """Test with single gpu."""
    model.eval()
    results = {}
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        batch_size = data['img'].shape[0]
        try:
            with torch.no_grad():
                cur_res = model(return_loss=False, **data)

            batch_size = len(cur_res)
            for k, v in cur_res.items():
                results[k] = v
        except:
            pass
        for _ in range(batch_size):
            prog_bar.update()
    return results


class BLADE_API():
    def __init__(
            self,
            batch_list,
            device,     # e.g. cuda:0
            samples_per_gpu,
            workers_per_gpu, # can be 8
            temp_output_folder=None,
            render_and_save_imgs=False,
            cfg='./blade/configs/blade_inthewild.py',
            # checkpoint='./networks/dav2_aios_redo_lessaug_morehummanpdhuman_deptheval_12x8_mbs7/epoch_2.pth',
            checkpoint='./pretrained/epoch_2.pth'
    ):
        if 'cuda' in device:
            # Extract the GPU index from the string, e.g., "cuda:1" -> "1"
            if ":" in device:
                gpu_id = device.split(":")[-1]
            else:
                gpu_id = 0
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cfg = os.path.abspath(os.path.join(self.base_dir, cfg)) \
                    if not os.path.isabs(cfg) else cfg
        self.cfg = mmcv.Config.fromfile(self.cfg)
        self.checkpoint = os.path.abspath(os.path.join(self.base_dir, checkpoint)) \
                            if not os.path.isabs(checkpoint) else checkpoint

        # init distributed env first, since logger depends on the dist info.
        # if args.launcher == 'none':
        #     distributed = False
        # else:
        #     distributed = True
        #     init_dist(args.launcher, **cfg.dist_params)

        # build the dataloader
        self.cfg.data.batch_list = batch_list
        if samples_per_gpu == 1:
            self.cfg.data.pipeline = inference_pipeline_batchof1
        dataset = build_dataset(self.cfg.data)
        # the extra round_up data will be removed during gpu/cpu collect
        self.data_loader = build_dataloader(dataset,
                                            samples_per_gpu=samples_per_gpu,
                                            workers_per_gpu=workers_per_gpu,
                                            dist=False,
                                            shuffle=False,
                                            round_up=False)
        # if args.demo_root:
        #     os.makedirs(args.demo_root, exist_ok=True)
            # cfg.model.visualizer.demo_root = args.demo_root

        # build the model and load checkpoint
        self.cfg.model.render_and_save_imgs = render_and_save_imgs
        self.cfg.model.temp_output_folder = os.path.abspath(os.path.join(self.base_dir, temp_output_folder)) \
                    if not os.path.isabs(temp_output_folder) else temp_output_folder
        os.makedirs(self.cfg.model.temp_output_folder, exist_ok=True)
        self.model = build_architecture(self.cfg.model)
        self.model.id_list = dataset.id_list
        self.model.demo = True
        self.model.vis = True
        self.model.miou = False
        self.model.pmiou = False

        # fp16_cfg = self.cfg.get('fp16', None)
        # if fp16_cfg is not None:
        #     wrap_fp16_model(self.model)
        print(f'BLADE: load weights from {self.checkpoint}')
        load_checkpoint(self.model, self.checkpoint, map_location='cpu')

        self.model = MMDataParallel(self.model, device_ids=[0])

    def process(self):
        print('Start single gpu test')
        outputs = single_gpu_test(self.model, self.data_loader)
        return outputs

