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

from blade.models.architectures.blade_imports import *





class BLADE(BaseArchitecture):

    def build_depthnet(self, depth_backbone_version, depth_head, depth_scale, depth_interface_size):
        # ------------------ Depth Network (DAv2) ---------------------
        self.depth_backbone_version = depth_backbone_version
        max_depth=20
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.depth_backbone = DepthAnythingV2(**{**model_configs[self.depth_backbone_version], 'max_depth': max_depth})
        self.depth_backbone.depth_head.forward = MethodType(forward, self.depth_backbone.depth_head)
        if depth_interface_size == 'large':
            print("Using Large Depth Interface")
            self.depth_interface = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        elif depth_interface_size == 'small':
            print("Using Small Depth Interface")
            self.depth_interface = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
        else:
            print("Using Regular Depth Interface")
            self.depth_interface = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        self.depth_head_cfg = depth_head
        self.depth_head = build_head(depth_head)
        self.depth_scale = depth_scale

    def load_depthnet(self, pretrained_depth_backbone_ckpt, depthnet_ckpt_path):
        self.pretrained_depth_backbone_ckpt = pretrained_depth_backbone_ckpt
        self.depthnet_ckpt_path = depthnet_ckpt_path
        if depthnet_ckpt_path is not None:
            depth_ckpt = torch.load(self.depthnet_ckpt_path)
            depth_head_dict = OrderedDict()
            depth_interface_dict = OrderedDict()
            depth_backbone_dict = OrderedDict()
            for k, v in depth_ckpt['state_dict'].items():
                k = k.replace('module.', '')
                if k.startswith('depth_head'):
                    depth_head_dict[k.replace('depth_head.', '')] = v
                elif k.startswith('depth_interface'):
                    depth_interface_dict[k.replace('depth_interface.', '')] = v
                elif k.startswith('depth_backbone'):
                    depth_backbone_dict[k.replace('depth_backbone.', '')] = v
            self.depth_head.load_state_dict(depth_head_dict, strict=True)
            self.depth_interface.load_state_dict(depth_interface_dict, strict=True)
            self.depth_backbone.load_state_dict(depth_backbone_dict, strict=True)
            print("loaded depth_head, depth_interface, depth_backbone")
        elif pretrained_depth_backbone_ckpt:
            # load pretrained DAv2, depth head is thus not trained
            self.depth_backbone.load_state_dict(torch.load(pretrained_depth_backbone_ckpt, map_location='cpu'))
        # self.depth_backbone.eval()
        # self.depth_interface.eval()
        # self.depth_head.eval()


    def build_posenet(self):
        # ----------------- Pose Network (ControlNet + AiOS) ----------------------
        from aios_repo.build_helper import get_args_parser
        parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
        __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
        aios_args = parser.parse_args(['-c', f'{root_dir}/aios_repo/config/aios_smplx_controlnet.py',
                                       '--options', 'backbone=resnet50', 'num_person=1', 'threshold=0.1',
                                       '--resume', f'{root_dir}/pretrained/model_init_weights/aios_checkpoint.pth',
                                       '--eval', '--inference'])
        # shutil.copy2(aios_args.config_file, f'{root_dir}/aios_repo/config/aios_smplx.py')
        from aios_repo.config.config import cfg as aios_cfg
        aios_cfg.merge_from_dict(aios_args.options)
        cfg_dict = aios_cfg._cfg_dict.to_dict()
        args_vars = vars(aios_args)
        for k, v in cfg_dict.items():
            if k not in args_vars:
                setattr(aios_args, k, v)
            else:
                continue
                raise ValueError('Key {} can used by args only'.format(k))
        model, criterion, postprocessors, _ = build_model_main(aios_args, aios_cfg)
        self.pose_backbone = model
        self.pose_backbone.aux_loss = False
        self.pose_postprocessors = postprocessors
        self.aios_args = aios_args
        self.aios_cfg = aios_cfg

    def load_posenet(self):
        try:
            print("Try loading pretrained AiOS weights...")
            checkpoint = torch.load(self.aios_args.resume, map_location='cpu')
            self.pose_backbone.load_state_dict(checkpoint['model'], strict=False)
            transformer_controlnet_dict = OrderedDict()
            input_proj_controlnet_dict = OrderedDict()
            backbone_controlnet_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                k = k.replace('module.', '')
                if k.startswith('transformer.'):
                    transformer_controlnet_dict[k.replace('transformer.', '')] = v
                elif k.startswith('input_proj'):
                    input_proj_controlnet_dict[k.replace('input_proj.', '')] = v
                elif k.startswith('backbone'):
                    backbone_controlnet_dict[k.replace('backbone.', '')] = v
            self.pose_backbone.transformer_controlnet.load_state_dict(transformer_controlnet_dict, strict=True)
            self.pose_backbone.input_proj_controlnet.load_state_dict(input_proj_controlnet_dict, strict=True)
            self.pose_backbone.backbone_controlnet.load_state_dict(backbone_controlnet_dict, strict=True)
            print("Done.")
        except:
            print("Couldn't load pretrained AiOS weights")

    def setup_posenet_gradient(self):
        # -------------- Freeze original copy & only train controlnet trainable copy -----------------
        for param in self.pose_backbone.parameters():
            param.requires_grad = False
        for controlnet_module in self.pose_backbone.trainable_modules:
            for param in controlnet_module.parameters():
                param.requires_grad = True

    def build_body_models_and_conversion(self):
        # ----------------- SMPL-X ------------------
        self.body_model_joint_num = self.pose_backbone.body_model_joint_num
        model_path = f'{dirname(dirname(dirname(dirname(abspath(__file__)))))}/body_models/'
        self.body_model_smplx_neutral = smplx.create(model_path, model_type='smplx', gender='neutral', use_pca=False,
                                                     flat_hand_mean=False)
        self.body_model_smplx_male = smplx.create(model_path, model_type='smplx', gender='male', use_pca=False,
                                                  flat_hand_mean=False)
        self.body_model_smplx_female = smplx.create(model_path, model_type='smplx', gender='female', use_pca=False,
                                                    flat_hand_mean=False)
        self.smplx_faces = torch.tensor(self.body_model_smplx_neutral.faces.astype(np.int))
        set_requires_grad(self.body_model_smplx_neutral, False)
        set_requires_grad(self.body_model_smplx_male, False)
        set_requires_grad(self.body_model_smplx_female, False)

        # ----------------- SMPL-X -> SMPL conversion ---------------
        smpl_repo_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '../../smplx_repo'))
        conversion_cfg, conversion_destination_model, conversion_def_matrix, conversion_mask_ids \
            = init_conversion(smpl_repo_path, self.depth_interface[0].weight.device, 'smplx', 'smpl')
        self.conversion_destination_model = conversion_destination_model
        self.conversion_def_matrix = conversion_def_matrix
        self.conversion_mask_ids = conversion_mask_ids
        self.conversion_cfg = conversion_cfg
        self.right_hip_idx = get_keypoint_idx('right_hip_extra', self.body_model_test.keypoint_dst)
        self.left_hip_idx = get_keypoint_idx('left_hip_extra', self.body_model_test.keypoint_dst)
        self.smpl_faces = self.body_model_train.faces_tensor.int().clone()

    def build_keypoint_detector(self, sapiens_config):
        # ------------------ SAPIENS ------------------
        # dummpy code, not using sapiens segmentation, but keep here to resolve mmcv scope issue for LayerNorm
        self.sapiens = build_sapiens(sapiens_config)
        del self.sapiens

        print(f'COCO wholebody -> Goliath mapping')
        for i, (coco_i, goliath_i) in enumerate(coco_wholebody_to_goliath_mapping.items()):
            print(f"{i} - COCO wholebody {coco_i} ('{cocowholebody_kpt_names[coco_i]['name']}') -> Goliath {goliath_i} ('{goliath_kpt_names[goliath_i]}')")
        self.coco_wholebody_in_goliath = torch.tensor(list(coco_wholebody_to_goliath_mapping.keys()))
        self.coco_wholebody_to_goliath_mapping_is_hand = ((self.coco_wholebody_in_goliath >= 92) & (self.coco_wholebody_in_goliath <= 111)) | \
                                                        ((self.coco_wholebody_in_goliath >= 113) & (self.coco_wholebody_in_goliath <= 132))

        self.kpt_mask = torch.ones((58))
        if self.ignore_face_kpts:
            self.kpt_mask[53:] = 0
            # self.kpt_mask[1:5] =
        if self.ignore_hand_kpts:
            self.kpt_mask[21:36] = self.kpt_mask[37:52] = 0
        if self.ignore_toe_kpts:
            self.kpt_mask[15:17] = self.kpt_mask[18:20] = 0

    def __init__(
        self,
        depth_head: Optional[Union[dict, None]] = None,
        uv_renderer: Optional[Union[dict, None]] = None,
        depth_renderer: Optional[Union[dict, None]] = None,
        resolution: int = 224,
        body_model_train: Optional[Union[dict, None]] = None,
        body_model_test: Optional[Union[dict, None]] = None,
        freeze_modules: Tuple[str] = (),
        ##
        loss_transl_z: Optional[Union[dict, None]] = None,
        ###
        init_cfg: Optional[Union[list, dict, None]] = None,
        # sapiens
        sapiens_config=None,
        # DepthAnything V2
        pretrained_depth_backbone_ckpt = None,
        depth_backbone_version = None,
        # AiOS
        depthnet_ckpt_path = None,
        joint_loss_weight=1.,
        vert_loss_weight=1.,
        do_res_aug=True,
        miou=False,
        pmiou=False,
        depth_scale=1.,
        do_stage_1=False,
        opt_pose=False,
        opt_tz=False,
        clear_background=False,
        convert_to_smpl=False,
        ignore_face_kpts=True,
        ignore_hand_kpts=True,
        ignore_toe_kpts=True,
        n_optimization_iterations=100,
        render_and_save_imgs=False,
        render_gt_instead=False,
        temp_output_folder=None,
        enable_vis_window=True,
        depth_interface_size=None,
        use_depth_map=False,
    ):
        if not enable_vis_window:
            import matplotlib
            matplotlib.use('Agg')

        super(BLADE, self).__init__(init_cfg)

        # ----------- INIT VARIABLES ------------
        self.freeze_modules = freeze_modules

        self.depth_renderer = build_renderer(depth_renderer)
        self.uv_renderer = build_renderer(uv_renderer)

        self.body_model_train = build_body_model(body_model_train)
        self.body_model_test = build_body_model(body_model_test)
        self.resolution = resolution

        self.loss_transl_z = build_loss(loss_transl_z)

        self.do_stage_1 = do_stage_1    # stage 1: depth only, stage 2: pose only
        self.opt_pose = opt_pose
        self.opt_tz = opt_tz
        self.clear_background = clear_background
        self.convert_to_smpl = convert_to_smpl
        self.ignore_face_kpts=ignore_face_kpts
        self.ignore_hand_kpts=ignore_hand_kpts
        self.ignore_toe_kpts=ignore_toe_kpts
        self.n_optimization_iterations = n_optimization_iterations
        self.render_and_save_imgs = render_and_save_imgs
        self.render_gt_instead = render_gt_instead
        self.temp_output_folder = temp_output_folder

        self.joint_loss_weight = joint_loss_weight
        self.vert_loss_weight = vert_loss_weight

        self.miou = miou
        self.pmiou = pmiou
        print(f'do miou is {miou}, do pmiou is {pmiou}')

        if do_res_aug:
            self.train_sizes = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        else:
            self.train_sizes = [480, 800]

        self.is_test_init_done = False

        # ----------- SETUP MODULES ------------
        # Depth Network (DAv2)
        self.build_depthnet(depth_backbone_version, depth_head, depth_scale, depth_interface_size)
        self.use_depth_map = use_depth_map
        if use_depth_map:
            print("Using Depth Map as Feature")

        # Pose Network (ControlNet + AiOS)
        self.build_posenet()

        # SMPL-X & conversion to SMPL
        self.build_body_models_and_conversion()

        # sapiens keypoint detector
        self.build_keypoint_detector(sapiens_config)


        # --------------------- LOAD WEIGHTS INTO THE MODULES ------------------------
        # load depthnet
        self.load_depthnet(pretrained_depth_backbone_ckpt, depthnet_ckpt_path)

        # load posenet
        self.load_posenet()


        # --------------------- GRADIENTS: FREEZE ORIGINAL, TRAIN CONTROLNET COPY ------------------------
        self.setup_posenet_gradient()


        # others
        self.global_rank = get_global_rank()
        self.local_rank = get_local_rank()

    def init_weights(self):
        super().init_weights()

        # Depth Anything V2
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        if self.depthnet_ckpt_path is not None:
            depth_ckpt = torch.load(self.depthnet_ckpt_path)
            depth_head_dict = OrderedDict()
            depth_interface_dict = OrderedDict()
            depth_backbone_dict = OrderedDict()
            for k, v in depth_ckpt['state_dict'].items():
                k = k.replace('module.', '')
                if k.startswith('depth_head'):
                    depth_head_dict[k.replace('depth_head.', '')] = v
                elif k.startswith('depth_interface'):
                    depth_interface_dict[k.replace('depth_interface.', '')] = v
                elif k.startswith('depth_backbone'):
                    depth_backbone_dict[k.replace('depth_backbone.', '')] = v
            self.depth_head.load_state_dict(depth_head_dict, strict=True)
            self.depth_interface.load_state_dict(depth_interface_dict, strict=True)
            self.depth_backbone.load_state_dict(depth_backbone_dict, strict=True)
            print("loaded depth_head, depth_interface, depth_backbone")
        elif self.pretrained_depth_backbone_ckpt:
            # load pretrained DAv2, depth head is thus not trained
            self.depth_backbone.load_state_dict(torch.load(self.pretrained_depth_backbone_ckpt, map_location='cpu'))
            print("loaded DAv2 to initialize depth_backbone")
        # self.depth_backbone.eval()
        # self.depth_interface.eval()
        # self.depth_head.eval()

        # AiOS
        checkpoint = torch.load(self.aios_args.resume, map_location='cpu')
        transformer_controlnet_dict = OrderedDict()
        input_proj_controlnet_dict = OrderedDict()
        backbone_controlnet_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            k = k.replace('module.', '')
            if k.startswith('transformer.'):
                transformer_controlnet_dict[k.replace('transformer.', '')] = v
            elif k.startswith('input_proj'):
                input_proj_controlnet_dict[k.replace('input_proj.', '')] = v
            elif k.startswith('backbone'):
                backbone_controlnet_dict[k.replace('backbone.', '')] = v
        self.pose_backbone.transformer_controlnet.load_state_dict(transformer_controlnet_dict, strict=True)
        self.pose_backbone.input_proj_controlnet.load_state_dict(input_proj_controlnet_dict, strict=True)
        self.pose_backbone.backbone_controlnet.load_state_dict(backbone_controlnet_dict, strict=True)


        # freeze original copy & train ControlNet copy
        for param in self.pose_backbone.parameters():
            param.requires_grad = False
        for controlnet_module in self.pose_backbone.trainable_modules:
            for param in controlnet_module.parameters():
                param.requires_grad = True


    def load_pretrained_depthnet(self):
        depth_ckpt = torch.load(self.depthnet_ckpt_path)
        depth_head_dict = OrderedDict()
        depth_interface_dict = OrderedDict()
        depth_backbone_dict = OrderedDict()
        for k, v in depth_ckpt['state_dict'].items():
            k = k.replace('module.', '')
            if k.startswith('depth_head'):
                depth_head_dict[k.replace('depth_head.', '')] = v
            elif k.startswith('depth_interface'):
                depth_interface_dict[k.replace('depth_interface.', '')] = v
            elif k.startswith('depth_backbone'):
                depth_backbone_dict[k.replace('depth_backbone.', '')] = v
        self.depth_head.load_state_dict(depth_head_dict, strict=True)
        self.depth_interface.load_state_dict(depth_interface_dict, strict=True)
        self.depth_backbone.load_state_dict(depth_backbone_dict, strict=True)
        print("loaded depth_head, depth_interface, depth_backbone")

    def visualize_smpl(self, data_batch, batch_size, device):
        smpl_body_pose = data_batch['smpl_body_pose']
        # smpl_body_pose = torch.cat([smpl_body_pose[:,:,:66], smpl_body_pose[:,:,-45:-42]],dim=2)[:,0]
        smpl_global_orient = data_batch['smpl_global_orient']
        smpl_betas = data_batch['smpl_betas'].float()
        origin_output = self.body_model_train(
            betas=smpl_betas[:,0],
            body_pose=smpl_body_pose[:,0].float(),
            global_orient=smpl_global_orient[:,0].float())
        smpl_vertices = origin_output['vertices']
        smpl_joints = origin_output['joints']
        smpl_pelvis = smpl_joints[:, :1]

        data_batch['gt_pelvis_3d'] = smpl_pelvis.clone()
        data_batch['gt_vert_3d_w_pelvis'] = smpl_vertices.clone()
        data_batch['gt_joints_3d_w_pelvis'] = smpl_joints.clone()
        data_batch['gt_vert_3d_no_pelvis'] = smpl_vertices - smpl_pelvis
        data_batch['gt_joints_3d_no_pelvis'] = smpl_joints - smpl_pelvis

        transl = data_batch['pelvis_camcoord'][:,None] - smpl_joints[:,0:1]

        gt_vert_homogeneous = torch.cat([smpl_vertices + transl,
                                         torch.ones_like(smpl_vertices[:, :, :1])], dim=-1).permute(0, 2, 1)
        gt_joints_homogeneous = torch.cat([smpl_joints + transl,
                                           torch.ones_like(smpl_joints[:, :, :1])], dim=-1).permute(0, 2, 1)

        # apply addition camera Rt if necessary
        gt_cam_K_posenet = data_batch['K']
        data_batch['gt_cam_K_posenet'] = gt_cam_K_posenet
        if 'cam_R' not in data_batch or 'cam_t' not in data_batch:
            gt_cam_Rt = torch.cat([torch.eye(3)[None].expand(batch_size, -1, -1), torch.zeros((batch_size, 3, 1))],
                                  dim=-1).to(device)
        else:
            gt_cam_Rt = torch.cat([data_batch['cam_R'], data_batch['cam_t']], dim=-1)

        gt_cam_Rt[:, 0, 0] *= -1
        gt_cam_Rt[:, 1, 1] *= -1

        # projection
        M_xform = gt_cam_K_posenet @ gt_cam_Rt
        gt_verts2d = torch.bmm(M_xform, gt_vert_homogeneous)
        gt_verts2d[:, 0] /= gt_verts2d[:, 2]
        gt_verts2d[:, 1] /= gt_verts2d[:, 2]

        gt_joints2d = torch.bmm(M_xform, gt_joints_homogeneous)
        gt_joints2d[:, 0] /= gt_joints2d[:, 2]
        gt_joints2d[:, 1] /= gt_joints2d[:, 2]

        pelvis = torch.cat([data_batch['pelvis_camcoord'][:,:,None], torch.ones_like(data_batch['pelvis_camcoord'][:,:,None])[:,:1]],dim=1)
        gt_pelvis2d = torch.bmm(M_xform, pelvis)
        gt_pelvis2d[:, 0] /= gt_pelvis2d[:, 2]
        gt_pelvis2d[:, 1] /= gt_pelvis2d[:, 2]

        data_batch['gt_cam_proj_xform'] = M_xform
        data_batch['gt_vert_2d'] = gt_verts2d.permute(0, 2, 1)
        data_batch['gt_joints_2d'] = gt_joints2d.permute(0, 2, 1)
        data_batch['gt_transl_w_pelvis'] = smpl_pelvis + data_batch['smpl_transl'][:, None]

        im_h, im_w = data_batch['orig_img'].shape[-2:]
        for plt_i in range(batch_size):
            plt.imshow(data_batch['orig_img'][plt_i].permute(1, 2, 0).detach().cpu());
            plt.scatter(gt_pelvis2d[plt_i, 0].detach().cpu(), gt_pelvis2d[plt_i, 1].detach().cpu(), s=4, c='r');
            # plt.scatter(gt_verts2d[plt_i, 0].detach().cpu(), gt_verts2d[plt_i, 1].detach().cpu(), c='r', marker='+', s=40);
            plt.title(f'pelvis depth={pelvis[plt_i,2,0]}m')
            plt.show()




    def train_step(self, data_batch, optimizer, **kwargs):
        """
        Stage 1 -- Trains DepthNet using groundtruth pelvis depth
                    data aug:      (1) feed data-augmented image to depthnet,
                               and (2) groundtruth pelvis coordinates only calculated for datasets that's not ours (which already has it calculated)

        Stage 2 -- Trains PoseNet using groundtruth SMPL-X
                    data aug:      (1) feed clean image to depthnet,
                               and (2) generate the SMPL-X models
        """


        device = data_batch['img'].device
        batch_size = data_batch['img'].shape[0]
        self.smplx_faces = self.smplx_faces.to(device)


        predictions = dict()
        for name in self.freeze_modules:
            for parameter in getattr(self, name).parameters():
                parameter.requires_grad = False



        # ------------------------------- Data Preparation (SMPL & SMPL-X) -------------------------------
        if self.do_stage_1:
            # our data has pelvis in camera coordinate already provided
            # the other datasets (H36M, ZOLLY, HuMMan, etc.) need conversion
            is_not_ourdata = torch.logical_not(data_batch['has_pelvis_camcoord'])
            if is_not_ourdata.any():
                batch_not_ourdata = {'smpl_body_pose': data_batch['smpl_body_pose'][is_not_ourdata],
                             'smpl_global_orient': data_batch['smpl_global_orient'][is_not_ourdata],
                             'smpl_betas': data_batch['smpl_betas'][is_not_ourdata],
                             'keypoints2d': data_batch['keypoints2d'][is_not_ourdata],
                             'smpl_transl': data_batch['smpl_transl'][is_not_ourdata]
                             }

                targets_not_ourdata = self.prepare_targets_depthonly(batch_not_ourdata)

                data_batch['has_pelvis_camcoord'][is_not_ourdata] = targets_not_ourdata['has_pelvis_camcoord']
                data_batch['pelvis_camcoord'] = data_batch['pelvis_camcoord'].float()
                data_batch['pelvis_camcoord'][is_not_ourdata] = targets_not_ourdata['pelvis_camcoord']
            targets = {'has_transl': data_batch['has_pelvis_camcoord'].clone(),
                       'smpl_transl': data_batch['pelvis_camcoord'].clone()}

            # self.visualize_smpl(data_batch, batch_size, device)

        else:
            targets = self.prepare_targets(data_batch)



        # (optional) convert data to half precision
        if torch.is_autocast_enabled():
            for key in data_batch.keys():
                if type(data_batch[key]) == torch.Tensor:
                    data_batch[key] = data_batch[key].half()


        # ------------------------------- Depth Prediction -------------------------------
        # extract intermediate feature from DepthAnythingV2 (DAV2)
        with torch.no_grad():
            self.depth_backbone.eval()
            x = data_batch['depthnet_img']
            patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
            if isinstance(self.depth_backbone, mmcv.parallel.distributed.MMDistributedDataParallel):
                depth_backbone = self.depth_backbone.module
            else:
                depth_backbone = self.depth_backbone
            features = depth_backbone.pretrained.get_intermediate_layers(x, depth_backbone.intermediate_layer_idx[depth_backbone.encoder],
                                                               return_class_token=True)
            depth_pred, depth_feat = depth_backbone.depth_head(features, patch_h, patch_w)
            depth_pred = depth_pred.detach()
            for f in depth_feat:
                f = f.detach()
            tmp_feat = resize(input=depth_feat[-1], size=(224, 224), mode="bilinear", align_corners=True)
            del depth_feat

        # predict pelvis depth Tz based on DAV2's feature
        with torch.set_grad_enabled(self.do_stage_1):
            if self.do_stage_1:
                self.depth_interface.train()
                self.depth_head.train()
            else:
                self.depth_interface.eval()
                self.depth_head.eval()
            if self.use_depth_map:
                tmp_feat[:,-1:] = resize(input=depth_pred, size=(224, 224), mode="bilinear", align_corners=True)
            dino_features = self.depth_interface(tmp_feat.clone())
            del tmp_feat
            predictions_verts = self.depth_head(dino_features)
            predictions.update(predictions_verts)



        # ------------------------------- Pose Prediction -----------------------------------
        if not self.do_stage_1:
            # prepare data
            # randomize resolution during training to improve robustness
            rand_res = random.choices(self.train_sizes, k=1)[0]
            posenet_input_img = F.interpolate(data_batch['posenet_img'], (rand_res, rand_res), mode="bilinear", align_corners=True)
            aios_data_batch = {'img': posenet_input_img,
                               'img_shape': torch.tensor([posenet_input_img.shape[-2],
                                                          posenet_input_img.shape[-1]], device=device)[None].repeat(batch_size, 1),
                               'body_bbox_center': [],
                               'body_bbox_size': [],
                               'pred_z': predictions['pred_z'].clone(),
                               'ann_idx': [torch.tensor([b_i], device=device) for b_i in range(batch_size)]
                               }
            # uses full image for pose estimator
            for b_i in range(batch_size):
                tmp = torch.tensor([0, 0, aios_data_batch['img'].shape[-1], aios_data_batch['img'].shape[-2]], device=device)
                aios_data_batch['body_bbox_center'].append(tmp.clone())
                aios_data_batch['body_bbox_size'].append(tmp.clone())

            # predict neutral SMPL-X pose
            self.pose_backbone.train()
            aios_outputs, aios_targets, aios_data_batch_nc  = self.pose_backbone(aios_data_batch)
            orig_target_sizes = torch.stack([t["size"] for t in aios_targets], dim=0)
            result, topk_smpl = self.pose_postprocessors['bbox'].forward_withgrad(aios_outputs, orig_target_sizes, aios_targets, aios_data_batch_nc)
            predictions['aios_outputs'] = aios_outputs

            # assuming there's only one person in the image, get the most confident prediction.
            # NOTE: should work with multi-person with minor adaptation
            all_expr, all_rhand_pose, all_lhand_pose, all_root_pose, all_pose, all_shape, all_pelvis \
                = [], [], [], [], [], [], []
            for b_i in range(batch_size):
                cur_result = result[b_i]
                all_shape.append(cur_result['smplx_shape'][0])
                all_pose.append(cur_result['smplx_body_pose'][0])
                all_root_pose.append(cur_result['smplx_root_pose'][0])
                all_lhand_pose.append(cur_result['smplx_lhand_pose'][0])
                all_rhand_pose.append(cur_result['smplx_rhand_pose'][0])
                all_expr.append(cur_result['smplx_expr'][0])
            all_shape = torch.stack(all_shape, dim=0)
            all_pose = torch.stack(all_pose, dim=0)
            all_root_pose = torch.stack(all_root_pose, dim=0)
            all_expr = torch.stack(all_expr, dim=0)
            all_lhand_pose = torch.stack(all_lhand_pose, dim=0)
            all_rhand_pose = torch.stack(all_rhand_pose, dim=0)
            predictions['pred_root_pose'] = all_root_pose
            predictions['pred_body_pose'] = all_pose
            predictions['pred_shape'] =all_shape

            # generate SMPL-X mesh based on predicted parameters
            zero_transl = torch.zeros((batch_size, 3), device=device)
            pred_output = self.body_model_smplx_neutral(betas=all_shape,
                                     body_pose=all_pose,
                                     left_hand_pose=all_lhand_pose,
                                     right_hand_pose=all_rhand_pose,
                                     global_orient=all_root_pose,
                                     transl=zero_transl,
                                     leye_pose=torch.zeros_like(zero_transl),
                                     reye_pose=torch.zeros_like(zero_transl),
                                     jaw_pose=torch.zeros_like(zero_transl),
                                     expression=all_expr,
                                     return_verts=True)
            pred_verts = pred_output.vertices
            pred_pelvis = pred_output.joints[:, :1]
            pred_joints = pred_output.joints

            predictions['pred_pelvis_3d'] = pred_pelvis.clone()
            predictions['pred_vert_3d_w_pelvis'] = pred_verts.clone()
            predictions['pred_joints_3d_w_pelvis'] = pred_joints.clone()
            predictions['pred_vert_3d_no_pelvis'] = pred_verts - pred_pelvis
            predictions['pred_joints_3d_no_pelvis'] = pred_joints - pred_pelvis

            # # visualize prediction
            # all_verts_no_pelvis = predictions['pred_vert_3d_no_pelvis'] + data_batch['gt_transl_w_pelvis']
            # all_joints_no_pelvis = predictions['pred_joints_3d_no_pelvis'] + data_batch['gt_transl_w_pelvis']
            # vert_homogeneous = torch.cat([all_verts_no_pelvis, torch.ones_like(all_verts_no_pelvis[:, :, :1])], dim=-1).permute(0, 2, 1)
            # joints_homogeneous = torch.cat([all_joints_no_pelvis, torch.ones_like(all_joints_no_pelvis[:, :, :1])], dim=-1).permute(0, 2, 1)
            # vert_proj = data_batch['gt_cam_proj_xform_posenet'] @ vert_homogeneous
            # vert_proj[:, 0] /= vert_proj[:, 2]
            # vert_proj[:, 1] /= vert_proj[:, 2]
            # joints_proj = data_batch['gt_cam_proj_xform_posenet'] @ joints_homogeneous
            # joints_proj[:, 0] /= joints_proj[:, 2]
            # joints_proj[:, 1] /= joints_proj[:, 2]
            # for plt_i in range(batch_size):
            #     plt.scatter(vert_proj[plt_i, 0].detach().cpu(), vert_proj[plt_i, 1].detach().cpu(), s=0.08);
            #     plt.scatter(joints_proj[plt_i, 0, :22].detach().cpu(), joints_proj[plt_i, 1, :22].detach().cpu(), c='r', marker='+', s=40);
            #     plt.imshow(data_batch['posenet_img'][plt_i].permute(1, 2, 0).detach().cpu() / 3 + 0.5);
            #     plt.show()
            #     plt.scatter(data_batch['gt_vert_2d_posenet'][plt_i, :, 0].detach().cpu(), data_batch['gt_vert_2d_posenet'][plt_i, :, 1].detach().cpu(), c='g', s=0.08);
            #     plt.scatter(data_batch['gt_joints_2d_posenet'][plt_i, :22, 0].detach().cpu(), data_batch['gt_joints_2d_posenet'][plt_i, :22, 1].detach().cpu(), c='r', marker='+', s=40);
            #     plt.imshow(data_batch['posenet_img'][plt_i].permute(1, 2, 0).detach().cpu() / 3 + 0.5);
            #     plt.show()


        # ------------------------------- Loss Calculation -----------------------------------
        losses = self.compute_losses(predictions, targets)
        for k, v in losses.items():
            if 'metric:' not in k:
                losses[k] = v.float()
        loss, log_vars = self._parse_losses(losses)

        if self.do_stage_1:
            if self.depth_interface is not None and 'depth_interface' in optimizer:
                optimizer['depth_interface'].zero_grad()
            if self.depth_head is not None and 'depth_head' in optimizer:
                optimizer['depth_head'].zero_grad()
        else:
            if self.pose_backbone is not None and 'pose_backbone' in optimizer:
                optimizer['pose_backbone'].zero_grad()

        loss.backward()

        if self.do_stage_1:
            if self.depth_interface is not None and 'depth_interface' in optimizer:
                optimizer['depth_interface'].step()
            if self.depth_head is not None and 'depth_head' in optimizer:
                optimizer['depth_head'].step()
        else:
            if self.pose_backbone is not None and 'pose_backbone' in optimizer:
                optimizer['pose_backbone'].step()

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(next(iter(data_batch.values()))))
        return outputs

    def prepare_targets_depthonly(self, data_batch: dict):
        gt_body_pose = data_batch['smpl_body_pose']
        gt_global_orient = data_batch['smpl_global_orient']
        gt_betas = data_batch['smpl_betas']
        gt_keypoints2d = data_batch['keypoints2d']

        tmp_output = self.body_model_train(
            betas=gt_betas.float(),
            body_pose=gt_body_pose.float(),
            global_orient=gt_global_orient.float(),
            num_joints=gt_keypoints2d.shape[1])
        tmp_model_joints = tmp_output['joints']

        if isinstance(self.body_model_train, mmcv.parallel.distributed.MMDistributedDataParallel):
            keypoint_dst = self.body_model_train.module.keypoint_dst
        else:
            keypoint_dst = self.body_model_train.keypoint_dst
        right_hip_idx = get_keypoint_idx('right_hip_extra', keypoint_dst)
        left_hip_idx = get_keypoint_idx('left_hip_extra', keypoint_dst)
        tmp_pelvis = (tmp_model_joints[:, right_hip_idx, :] +
                      tmp_model_joints[:, left_hip_idx, :]) / 2
        data_batch['pelvis_camcoord'] = tmp_pelvis + data_batch['smpl_transl']
        data_batch['has_pelvis_camcoord'] = torch.ones_like(data_batch['pelvis_camcoord'][:,0], dtype=torch.long)
        # data_batch['depthnet_img'] = data_batch['img']
        return data_batch

    def prepare_targets(self, data_batch: dict):

        device = data_batch['img'].device
        batch_size = data_batch['img'].shape[0]


        if 'has_SMPLX' in data_batch and data_batch['has_SMPLX'].view(-1).any():

            # SMPL-X
            # NOTE: the pelvis here are WITHOUT translation!
            data_batch['SMPLX_orig_pelvis'] = []
            data_batch['gt_pelvis_3d'] = []
            data_batch['gt_vert_3d_w_pelvis'] = []
            data_batch['gt_joints_3d_w_pelvis'] = []
            data_batch['gt_vert_3d_no_pelvis'] = []
            data_batch['gt_joints_3d_no_pelvis'] = []

            for b_i in range(batch_size):
                # get original pelvis offset before the parameters were changed/augmented
                orig_betas = data_batch['SMPLX_betas'][b_i, None]
                orig_transl = torch.zeros((1, 3), device=device)
                orig_body_pose = data_batch['SMPLX_body_pose_orig'][b_i, None]
                orig_left_hand_pose = data_batch['SMPLX_left_hand_pose_orig'][b_i, None]
                orig_right_hand_pose = data_batch['SMPLX_right_hand_pose_orig'][b_i, None]
                orig_global_orient = data_batch['SMPLX_global_orient_orig'][b_i, None]

                if 'SMPLX_gender' not in data_batch or data_batch['SMPLX_gender'][b_i] == 0:
                    body_model = self.body_model_smplx_neutral
                elif data_batch['SMPLX_gender'][b_i] > 0:
                    body_model = self.body_model_smplx_male
                elif data_batch['SMPLX_gender'][b_i] < 0:
                    body_model = self.body_model_smplx_female
                else:
                    assert False, f'Wrong gender value: {data_batch["SMPLX_gender"][b_i]}'

                orig_output = body_model(betas=orig_betas, body_pose=orig_body_pose,
                                         left_hand_pose=orig_left_hand_pose, right_hand_pose=orig_right_hand_pose,
                                         global_orient=orig_global_orient, transl=orig_transl,
                                         leye_pose=torch.zeros_like(orig_transl),
                                         reye_pose=torch.zeros_like(orig_transl),
                                         jaw_pose=torch.zeros_like(orig_transl),
                                         expression=torch.zeros_like(orig_betas),
                                         return_verts=True)
                data_batch['SMPLX_orig_pelvis'].append(orig_output.joints[:, :1])


                # get augmented params
                gt_betas = data_batch['SMPLX_betas'][b_i, None]
                zero_transl = torch.zeros((1, 3), device=device)
                gt_body_pose = data_batch['SMPLX_body_pose'][b_i, None]
                gt_left_hand_pose = data_batch['SMPLX_left_hand_pose'][b_i, None]
                gt_right_hand_pose = data_batch['SMPLX_right_hand_pose'][b_i, None]
                gt_global_orient = data_batch['SMPLX_global_orient'][b_i, None]
                output = body_model(betas=gt_betas, body_pose=gt_body_pose,
                                         left_hand_pose=gt_left_hand_pose, right_hand_pose=gt_right_hand_pose,
                                         global_orient=gt_global_orient, transl=zero_transl,
                                         leye_pose=torch.zeros_like(zero_transl),
                                         reye_pose=torch.zeros_like(zero_transl),
                                         jaw_pose=torch.zeros_like(zero_transl),
                                         expression=torch.zeros_like(gt_betas),
                                         return_verts=True)
                gt_vertices = output.vertices
                gt_pelvis = output.joints[:, :1]
                # gt_joints = output.joints[:, 1:1+21]
                gt_joints = output.joints

                data_batch['gt_pelvis_3d'].append(gt_pelvis.clone())
                data_batch['gt_vert_3d_w_pelvis'].append(gt_vertices.clone())
                data_batch['gt_joints_3d_w_pelvis'].append(gt_joints.clone())
                data_batch['gt_vert_3d_no_pelvis'].append(gt_vertices - gt_pelvis)
                data_batch['gt_joints_3d_no_pelvis'].append(gt_joints - gt_pelvis)
            data_batch['SMPLX_orig_pelvis'] = torch.cat(data_batch['SMPLX_orig_pelvis'], dim=0)
            data_batch['gt_pelvis_3d'] = torch.cat(data_batch['gt_pelvis_3d'], dim=0)
            data_batch['gt_vert_3d_w_pelvis'] = torch.cat(data_batch['gt_vert_3d_w_pelvis'], dim=0)
            data_batch['gt_joints_3d_w_pelvis'] = torch.cat(data_batch['gt_joints_3d_w_pelvis'], dim=0)
            data_batch['gt_vert_3d_no_pelvis'] = torch.cat(data_batch['gt_vert_3d_no_pelvis'], dim=0)
            data_batch['gt_joints_3d_no_pelvis'] = torch.cat(data_batch['gt_joints_3d_no_pelvis'], dim=0)

            # get translation for augmented data (flipped & rotated)
            flip_flag = data_batch['flip_flag']
            cam_transl = data_batch['SMPLX_transl'][:, None] + data_batch['SMPLX_orig_pelvis']
            cam_transl[:,:,0] *= flip_flag
            cam_transl = (data_batch['SMPLX_rotation_aug'].transpose(-2,-1) @ cam_transl.transpose(-2,-1)).transpose(-2,-1)
            data_batch['pelvis_camcoord'] = data_batch['gt_transl_w_pelvis'] = cam_transl

            # get 2D projection
            gt_vert_homogeneous = torch.cat([data_batch['gt_vert_3d_no_pelvis'] + data_batch['gt_transl_w_pelvis'],
                                             torch.ones_like(data_batch['gt_vert_3d_no_pelvis'][:,:,:1])], dim=-1).permute(0, 2, 1)
            gt_joints_homogeneous = torch.cat([data_batch['gt_joints_3d_no_pelvis'] + data_batch['gt_transl_w_pelvis'],
                                               torch.ones_like(data_batch['gt_joints_3d_no_pelvis'][:,:,:1])], dim=-1).permute(0, 2, 1)

            gt_cam_K = data_batch['K']
            data_batch['gt_cam_K'] = gt_cam_K
            if 'cam_R' not in data_batch or 'cam_t' not in data_batch:
                gt_cam_Rt = torch.cat([torch.eye(3)[None].expand(batch_size, -1, -1),torch.zeros((batch_size, 3, 1))], dim=-1).to(device)
            else:
                gt_cam_Rt = torch.cat([data_batch['cam_R'],data_batch['cam_t']], dim=-1)

            gt_cam_K_posenet = data_batch['posenet_K']
            data_batch['gt_cam_K_posenet'] = gt_cam_K_posenet

            M_xform_posenet = gt_cam_K_posenet @ gt_cam_Rt
            gt_verts2d_posenet = torch.bmm(M_xform_posenet, gt_vert_homogeneous)
            gt_verts2d_posenet[:, 0] /= gt_verts2d_posenet[:, 2]
            gt_verts2d_posenet[:, 1] /= gt_verts2d_posenet[:, 2]

            gt_pelvis2d_posenet = torch.bmm(M_xform_posenet, gt_joints_homogeneous)
            gt_pelvis2d_posenet[:, 0] /= gt_pelvis2d_posenet[:, 2]
            gt_pelvis2d_posenet[:, 1] /= gt_pelvis2d_posenet[:, 2]

            data_batch['gt_cam_proj_xform_posenet'] = M_xform_posenet
            data_batch['gt_vert_2d_posenet'] = gt_verts2d_posenet.permute(0, 2, 1)
            data_batch['gt_joints_2d_posenet'] = gt_pelvis2d_posenet.permute(0, 2, 1)
            # for plt_i in range(batch_size):
            #     plt.scatter(gt_verts2d_posenet[plt_i, 0].detach().cpu(), gt_verts2d_posenet[plt_i, 1].detach().cpu(), s=0.02);
            #     plt.scatter(gt_pelvis2d_posenet[plt_i, 0].detach().cpu(), gt_pelvis2d_posenet[plt_i, 1].detach().cpu(), c='r', marker='+', s=40);
            #     plt.imshow(data_batch['posenet_img'][plt_i].permute(1, 2, 0).detach().cpu() / 3 + 0.6);
            #     plt.title(f'pelvis depth: {data_batch["pelvis_camcoord"][plt_i,0,2]:.3f}')
            #     plt.show()
        else:

            # SMPL, evaluation only

            # get 3D SMPL mesh
            smpl_body_pose = data_batch['smpl_body_pose']
            smpl_global_orient = data_batch['smpl_global_orient']
            smpl_betas = data_batch['smpl_betas'].float()
            origin_output = self.body_model_train(
                betas=smpl_betas,
                body_pose=smpl_body_pose.float(),
                global_orient=smpl_global_orient.float())
            smpl_vertices = origin_output['vertices']
            smpl_joints = origin_output['joints']
            smpl_pelvis = smpl_joints[:, :1]

            data_batch['gt_pelvis_3d'] = smpl_pelvis.clone()
            data_batch['gt_vert_3d_w_pelvis'] = smpl_vertices.clone()
            data_batch['gt_joints_3d_w_pelvis'] = smpl_joints.clone()
            data_batch['gt_vert_3d_no_pelvis'] = smpl_vertices - smpl_pelvis
            data_batch['gt_joints_3d_no_pelvis'] = smpl_joints - smpl_pelvis

            # get 2D projection
            gt_vert_homogeneous = torch.cat([smpl_vertices + data_batch['smpl_transl'][:, None],
                                             torch.ones_like(smpl_vertices[:,:,:1])], dim=-1).permute(0, 2, 1)
            gt_joints_homogeneous = torch.cat([smpl_joints + data_batch['smpl_transl'][:, None],
                                               torch.ones_like(smpl_joints[:,:,:1])], dim=-1).permute(0, 2, 1)

            gt_cam_K_posenet = data_batch['posenet_K']
            data_batch['gt_cam_K_posenet'] = gt_cam_K_posenet
            if 'cam_R' not in data_batch or 'cam_t' not in data_batch:
                gt_cam_Rt = torch.cat([torch.eye(3)[None].expand(batch_size, -1, -1),torch.zeros((batch_size, 3, 1))], dim=-1).to(device)
            else:
                gt_cam_Rt = torch.cat([data_batch['cam_R'],data_batch['cam_t']], dim=-1)

            # projection
            M_xform = gt_cam_K_posenet @ gt_cam_Rt
            gt_verts2d = torch.bmm(M_xform, gt_vert_homogeneous)
            gt_verts2d[:, 0] /= gt_verts2d[:, 2]
            gt_verts2d[:, 1] /= gt_verts2d[:, 2]

            gt_joints2d = torch.bmm(M_xform, gt_joints_homogeneous)
            gt_joints2d[:, 0] /= gt_joints2d[:, 2]
            gt_joints2d[:, 1] /= gt_joints2d[:, 2]

            data_batch['gt_cam_proj_xform'] = M_xform
            data_batch['gt_vert_2d'] = gt_verts2d.permute(0, 2, 1)
            data_batch['gt_joints_2d'] = gt_joints2d.permute(0, 2, 1)
            data_batch['pelvis_camcoord'] = data_batch['gt_transl_w_pelvis'] = smpl_pelvis + data_batch['smpl_transl'][:, None]


            # for plt_i in range(batch_size):
            #     plt.imshow(data_batch['posenet_img'][plt_i].permute(1, 2, 0).detach().cpu() / 2.6 + 0.5);
            #     plt.scatter(gt_verts2d[plt_i, 0].detach().cpu(), gt_verts2d[plt_i, 1].detach().cpu(), s=0.02);
            #     plt.scatter(gt_joints2d[plt_i, 0].detach().cpu(), gt_joints2d[plt_i, 1].detach().cpu(), c='r', marker='+', s=40);
            #     plt.show()

        return data_batch

    def compute_losses(self, predictions: dict, targets: dict):
        """Compute losses."""

        losses = {}

        #  Tz
        if self.do_stage_1 and self.loss_transl_z is not None:
            pred_z = predictions['pred_z']
            has_transl = targets['has_transl']
            gt_transl = targets['smpl_transl']
            transl_weight = 1. / gt_transl[:, 2] * has_transl.view(-1)
            losses['transl_loss'] = self.compute_transl_loss(
                pred_z, gt_transl[..., 2:3], transl_weight)


        # NOTE: smpl-x  !!!!!!!!!!!!!!  pelvis not removed yet  !!!!!!!!!!!!!!
        if not self.do_stage_1:
            losses['metric:Tz_error'] = (predictions['pred_z'] - targets['smpl_transl'][:,-1:]).abs().mean()

            bs = predictions['pred_root_pose'].shape[0]

            # root pose
            pred_root_ori_mat = axis_angle_to_matrix(predictions['pred_root_pose']).float()
            gt_root_ori_mat = axis_angle_to_matrix(targets['SMPLX_global_orient'][:, 0]).float()
            losses['root_loss'] = so3_relative_angle(pred_root_ori_mat, gt_root_ori_mat).mean()

            # body pose
            pred_body_pose_mat = axis_angle_to_matrix(predictions['pred_body_pose'].view(bs, -1, 3))
            gt_body_pose_mat = axis_angle_to_matrix(targets['SMPLX_body_pose'])
            losses['body_pose_loss'] = so3_relative_angle(pred_body_pose_mat.flatten(0,1), gt_body_pose_mat.flatten(0,1)).mean()

            # shape
            losses['shape_loss'] = F.l1_loss(predictions['pred_shape'], targets['SMPLX_betas'])

            # body joints
            pred = predictions['pred_joints_3d_no_pelvis'][:,:self.body_model_joint_num]
            gt = targets['gt_joints_3d_no_pelvis'][:,:self.body_model_joint_num]
            losses['body_joint_loss'] = self.joint_loss_weight * F.l1_loss(pred, gt)

            losses['metric:mpjpe_all_body_joints'] = (((pred) - (gt))**2).sum(dim=-1).sqrt().mean()

            # vertices
            pred = predictions['pred_vert_3d_no_pelvis']
            gt = targets['gt_vert_3d_no_pelvis']
            losses['vert_loss'] = self.vert_loss_weight * F.l1_loss(pred, gt)
            losses['metric:mean_vert_error'] = (((pred) - (gt))**2).sum(dim=-1).sqrt().mean()

        return losses

    def init_detectors(self, device):
        if not hasattr(self, 'pose_model_2d'):
            self.pose_model_2d = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True)
        if not hasattr(self, 'kpt_estimator'):
            parser = get_sapiens_kpts_config()
            DATASET = 'goliath'
            MODEL_NAME = 'sapiens_1b'
            MODEL = f"{MODEL_NAME}-210e_{DATASET}-1024x768"
            DETECTION_CONFIG_FILE = f'{root_dir}/sapiens/pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py'
            DETECTION_CHECKPOINT = f'{root_dir}/pretrained/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
            CONFIG_FILE = f"{root_dir}/sapiens/pose/configs/sapiens_pose/{DATASET}/{MODEL}.py"
            CHECKPOINT = f'{root_dir}/pretrained/pose/{MODEL_NAME}_{DATASET}_best_goliath_AP_639.pth'
            KPT_THRES = 0.3  ## default keypoint confidence
            LINE_THICKNESS = 3  ## line thickness of the skeleton
            RADIUS = 3  ## keypoint radius
            self.kpt_args = parser.parse_args([f'{DETECTION_CONFIG_FILE}', f'{DETECTION_CHECKPOINT}', f'{CONFIG_FILE}',
                                           f'{CHECKPOINT}', f'--radius', f'{RADIUS}',
                                           '--kpt-thr', f'{KPT_THRES}', '--thickness', f'{LINE_THICKNESS}'])

            # build detector
            self.bbox_detector = init_detector(self.kpt_args.det_config, self.kpt_args.det_checkpoint, device=device)
            self.bbox_detector.cfg = adapt_mmdet_pipeline(self.bbox_detector.cfg)

            # build pose estimator
            self.kpt_estimator = init_pose_estimator(
                self.kpt_args.pose_config,
                self.kpt_args.pose_checkpoint,
                override_ckpt_meta=True,  # dont load the checkpoint meta data, load from config file
                device=device,
                cfg_options=dict(
                    model=dict(test_cfg=dict(output_heatmaps=self.kpt_args.draw_heatmap))))

            # build segmenter
            # MODEL = f"{MODEL_NAME}_{DATASET}-1024x768"
            # CHECKPOINT = f'{root_dir}/sapiens/checkpoints/seg/{MODEL_NAME}_{DATASET}_best_goliath_mIoU_7994_epoch_151.pth'
            # CONFIG_FILE = f"{root_dir}/sapiens/seg/configs/sapiens_seg/{DATASET}/{MODEL}.py"
            # self.segmenter = init_model(CONFIG_FILE, CHECKPOINT, device=device)

    def get_gt_seg_mask(self, data_batch):
        """ render groundtruth segmentation mask"""
        test_res = self.resolution
        K = data_batch['K']
        gt_transl = data_batch['smpl_transl']
        ori_shape = data_batch['ori_shape']
        ori_focal_length = data_batch['ori_focal_length'].float().view(-1)
        px = ori_shape[:, 1].float() / 2
        py = ori_shape[:, 0].float() / 2
        has_K_ids = torch.where(data_batch['has_K'] == 1)[0]
        px[has_K_ids] = K[has_K_ids, 0, 2].float()
        py[has_K_ids] = K[has_K_ids, 1, 2].float()
        gt_body_pose = data_batch['smpl_body_pose']
        gt_global_orient = data_batch['smpl_global_orient']
        gt_betas = data_batch['smpl_betas'].float()
        gt_output = self.body_model_test(
            betas=gt_betas,
            body_pose=gt_body_pose.float(),
            global_orient=gt_global_orient.float())
        gt_vertices = gt_output['vertices']
        gt_joints = gt_output['joints']
        gt_pelvis = gt_output['joints']

        gt_mask_h, gt_mask_w = data_batch['posenet_img'].shape[-2:]
        gt_mask_w_gt = self.render_segmask(
            vertices=gt_vertices,
            transl=gt_transl,
            center=data_batch['center'].float(),
            scale=data_batch['scale'][:, 0].float(),
            focal_length_ndc=ori_focal_length.float() / data_batch['scale'][:, 0].float() * 2,
            px=data_batch['center'][:, 0],
            py=data_batch['center'][:, 1],
            img_res=max(gt_mask_h, gt_mask_w))
        return gt_mask_w_gt, gt_vertices


    def detect_seg_mask(self, images, gt_mask_w_gt):
        """ generate segmentation mask
            if
        """
        batch_size = images.shape[0]
        device = images.device

        seg_masks = []
        seged_img = []
        has_seg_mask = []
        for i in range(batch_size):
            mp_input = images[i].permute(1, 2, 0).cpu().numpy()
            mp_results = self.pose_model_2d.process(mp_input)

            if mp_results.segmentation_mask is not None:
                # Value in [0, 1] that says how "tight" to make the segmentation. Greater => tighter
                tightness = .2
                if self.clear_background:
                    condition = np.stack((mp_results.segmentation_mask,) * 3, axis=-1) > tightness
                    bg_image = np.zeros(mp_input.shape, dtype=np.uint8)
                    bg_image[:] = [255, 255, 255]
                    annotated_image = np.where(condition, mp_input.copy(), bg_image)
                else:
                    condition = (images[i].sum(0) > 0)[:, :, None].expand(-1, -1, 3).cpu().numpy()
                    annotated_image = mp_input.copy()

                seg_masks.append(torch.tensor(condition[:, :, 0:1], device=device))
                seged_img.append(torch.tensor(annotated_image, device=device))
                has_seg_mask.append(True)
            else:
                if self.is_demo:
                    assert False, "failed to segment the human, thus cannot solve for camera, skipping this batch"
                print("no seg mask, using GT mask if available")
                seg_masks.append((gt_mask_w_gt[i].permute(1,2,0)>0)*1.)
                seged_img.append(torch.tensor(mp_input).to(device))
                has_seg_mask.append(False)
                # plt.imshow(seged_img[-1]); plt.show()
        seg_masks = torch.stack(seg_masks, dim=0).permute(0, 3, 1, 2)
        seged_img = torch.stack(seged_img, dim=0).permute(0, 3, 1, 2)
        orig_img = images.clone()

        return seg_masks, seged_img, orig_img, has_seg_mask


    def get_rtm_bbox(self, rtm_input):
        # rtm_input = data_batch['orig_img'].cpu().numpy() * 1.
        batch_size = rtm_input.shape[0]
        rtm_input = rtm_input.transpose(0, 2, 3, 1)
        rtm_input_list = [rtm_input[i] for i in range(batch_size)]

        det_result = inference_detector(self.bbox_detector, rtm_input_list)
        bboxes_list = []
        for i in range(batch_size):
            pred_instance = det_result[i].pred_instances.cpu().numpy()
            bboxes = np.concatenate(
                (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
            bboxes = bboxes[np.logical_and(pred_instance.labels == self.kpt_args.det_cat_id,
                                           pred_instance.scores > self.kpt_args.bbox_thr)]
            bboxes = bboxes[nms(bboxes, self.kpt_args.nms_thr), :4]
            if len(bboxes) == 0:
                # NOTE: if no box, use mediapipe's box
                bboxes_list.append(np.array([[0, 0, rtm_input.shape[-3], rtm_input.shape[-2]]]))
            else:
                bboxes_list.append(bboxes[:1])

            # plt.imshow(rtm_input_list[i]/255);
            # x_min, y_min, width, height = bboxes[0]
            # rect = plt.Rectangle((x_min, y_min), width, height, edgecolor='red', facecolor='none', linewidth=2)
            # plt.gca().add_patch(rect); plt.show()

        bboxes_list = np.stack(bboxes_list, axis=0)

        return bboxes_list

    def get_padding_mask(self, data_batch, downscale):
        N, _, H, W = data_batch['posenet_img'].shape
        device = data_batch['posenet_img'].device

        # Create a mask with ones, which will mark the padded regions.
        mask = torch.ones((N, 1, H, W), device=device)

        # For each image, calculate the offsets where the original image sits.
        # Here we assume the padding is symmetric (i.e. the original image is centered).
        for i in range(N):
            h, w = data_batch['ori_shape'][i]
            scale = H / max(h.item(), w.item())
            h = int((h * scale).floor().item())
            w = int((w * scale).floor().item())

            pad_top = (H - h) // 2
            pad_left = (W - w) // 2
            # Set the region corresponding to the original image to 0
            mask[i, 0, pad_top:pad_top + h, pad_left:pad_left + w] = 0.

        mask = F.interpolate(mask, (H// downscale, W// downscale), mode="bilinear", align_corners=True)

        data_batch['pad_mask'] = mask

    def solve_fTxTy_pytorch3d(self, all_preds, data_batch, predictions):
        pose_img_h, pose_img_w = data_batch['posenet_img'].shape[-2:]
        out_hw = (pose_img_h, pose_img_w)

        ktps_list, orig_good_flag, seg_mask, seged_img, has_seg_mask, smpl_verts, input_T, input_f \
            = (all_preds['ktps_list'], all_preds['good_flag'], all_preds['seg_masks'], all_preds['seged_img'],
               all_preds['has_seg_mask'], all_preds['vertices'], all_preds['pred_transl'], all_preds['pred_f'])
        if not self.convert_to_smpl:
            good_flag = (self.kpt_mask[None]>0) & orig_good_flag
        else:
            good_flag = orig_good_flag
        batch_size, n_pts = smpl_verts.shape[0], smpl_verts.shape[1]
        device = smpl_verts.device


        # if self.convert_to_smpl:
        #     ktps_list = ktps_list[:,:,goliath58_to_coco17]
        #     good_flag = good_flag[:,goliath58_to_coco17]

        target_kpts_2d = ktps_list.clone().detach()
        normalize_scale = torch.max(target_kpts_2d[:, 0].max(-1)[0] - target_kpts_2d[:, 0].min(-1)[0],
                                    target_kpts_2d[:, 1].max(-1)[0] - target_kpts_2d[:, 1].min(-1)[0])

        init_T = input_T.detach().clone(); init_f = input_f.detach().clone()

        # Define output rasterization size
        downscale = 1
        if max(seg_mask.shape) > 512:
            downscale *= 2
        out_h, out_w = out_hw[0] // downscale, out_hw[1] // downscale  # Output raster size
        principal_point = torch.zeros((batch_size, 2), requires_grad=False, device=device)
        principal_point[:, 0] = out_w / 2
        principal_point[:, 1] = out_h / 2

        seg_mask_down = F.interpolate(seg_mask * 1., (out_h, out_w), mode="bilinear", align_corners=True)
        if batch_size > 1:
            self.get_padding_mask(data_batch, downscale)
            seg_mask_down = (seg_mask_down + data_batch['pad_mask']).clamp(0,1)

        # Prepare Meshes object
        textures = torch.ones((batch_size, n_pts, 3), dtype=torch.float32, device=device)  # White color
        textures = TexturesVertex(verts_features=textures)  # [bs, n_pts, 3]

        # Define PyTorch3D renderer with Rasterizer and Shader
        blend_params = BlendParams(background_color=(0.0, 0.0, 0.0), sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=(out_h, out_w),
            # blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma * 4,
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma * 0.01,
            faces_per_pixel=2,
            bin_size=0,  # avoid bin size warning, but slower
            # max_faces_per_bin=200,  # You can also try increasing this value
            cull_backfaces=True,
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=None,  # We'll define cameras per iteration
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(
                blend_params=blend_params  # Black background
            )
        )  # [bs, 3]

        if self.convert_to_smpl:
            orig_betas = predictions['converted_smpl_betas'].detach().clone()
            orig_body_pose = predictions['converted_smpl_body_pose'].detach().clone()
            orig_global_orient = predictions['converted_smpl_global_orient'].detach().clone()
            # NOTE: init_T is calculated with this transl, they should be merged
            orig_transl = predictions['converted_smpl_transl'].detach().clone()
            faces = self.smpl_faces[None].repeat(batch_size, 1, 1).to(device)
        else:
            orig_betas = predictions['pred_shape'].detach().clone()
            orig_body_pose = predictions['pred_body_pose'].detach().clone()
            orig_global_orient = predictions['pred_root_pose'].detach().clone()
            orig_lhand_pose = predictions['pred_lhand_pose'].detach().clone()
            orig_rhand_pose = predictions['pred_rhand_pose'].detach().clone()
            # NOTE: init_T is calculated with this transl, they should be merged
            orig_transl = torch.zeros((batch_size, 3), device=device)
            faces = self.smplx_faces[None].repeat(batch_size, 1, 1).to(device)

        with ((torch.set_grad_enabled(True))):
            f = init_f[:, None].clone().float(); f.requires_grad = True
            Tx = init_T[:, 0]; Tx.requires_grad = True
            Ty = init_T[:, 1]; Ty.requires_grad = True
            # fixed_Tz = init_T[:, 2].clone().detach(); fixed_Tz.requires_grad = False
            Tz = init_T[:, 2].clone().detach(); Tz.requires_grad = True
            opti_betas = orig_betas.clone().detach(); opti_betas.requires_grad = True
            opti_body_pose = orig_body_pose.clone().detach(); opti_body_pose.requires_grad = True
            opti_global_orient = orig_global_orient.clone().detach(); opti_global_orient.requires_grad = True
            if not self.convert_to_smpl:
                opti_lhand_pose = orig_lhand_pose.clone().detach(); opti_lhand_pose.requires_grad = True
                opti_rhand_pose = orig_rhand_pose.clone().detach(); opti_rhand_pose.requires_grad = True


            # TODO: this transl is already merged with init_T
            opti_transl = orig_transl.clone().detach(); opti_transl.requires_grad = False

            # Define optimizer
            num_iterations = self.n_optimization_iterations

            # for 2D alignment, with pose shape ori optimization
            optimizer = optim.SGD([
                {'params': [Tx, Ty], 'lr': 0.002},
                {'params': [f], 'lr': 0.01 * max(pose_img_h, pose_img_w)},   # focal length need to change freely to handle close up cases
                {'params': [opti_global_orient], 'lr': 0.01 if self.opt_pose else 0.},
                {'params': [opti_body_pose], 'lr': 0.01 if self.opt_pose else 0.},      # body pose should change very minimally
                {'params': [Tz], 'lr': 0.05 if self.opt_tz else 0.}
            ], momentum=0.9)
            if not self.convert_to_smpl:
                optimizer.add_param_group({'params': opti_lhand_pose, 'lr': 0.01})
                optimizer.add_param_group({'params': opti_rhand_pose, 'lr': 0.01})

            for iteration in range(num_iterations):
                optimizer.zero_grad()

                T = torch.cat([Tx, Ty, Tz], dim=1)

                # new SMPL
                if self.convert_to_smpl:
                    origin_output = self.body_model_test(
                        betas=opti_betas,
                        body_pose=opti_body_pose.float(),
                        global_orient=opti_global_orient.float(),
                        transl=opti_transl)
                    opti_smpl_verts = origin_output['vertices']
                    if isinstance(self.depth_backbone, mmcv.parallel.distributed.MMDistributedDataParallel):
                        body_model = self.body_model_train.module
                    else:
                        body_model = self.body_model_train
                    opti_joints_coco, _ = convert_kps(body_model.forward_joints(dict(vertices=opti_smpl_verts))[0],
                                                           body_model.keypoint_src, 'coco', approximate=False)
                    verts = opti_smpl_verts + T[:, None]
                    meshes = Meshes(verts=verts, faces=faces, textures=textures)
                else:
                    zero_like_transl = torch.zeros_like(opti_transl)
                    pred_output = self.body_model_smplx_neutral(betas=opti_betas,
                                                  body_pose=opti_body_pose,
                                                  left_hand_pose=opti_lhand_pose,
                                                  right_hand_pose=opti_rhand_pose,
                                                  global_orient=opti_global_orient,
                                                  transl=opti_transl,
                                                  leye_pose=zero_like_transl,
                                                  reye_pose=zero_like_transl,
                                                  jaw_pose=zero_like_transl,
                                                  expression=torch.zeros_like(opti_betas).detach(),
                                                  return_verts=True)
                    opti_smpl_verts = pred_output.vertices
                    opti_joints_smplx = F.pad(pred_output.joints, (0, 0, 0, 17), mode='constant', value=0)
                    opti_joints_coco_wholebody, _ = convert_kps(keypoints=opti_joints_smplx, src='smplx', dst='coco_wholebody')
                    opti_joints_coco = opti_joints_coco_wholebody[:, self.coco_wholebody_in_goliath]
                    verts = opti_smpl_verts + T[:, None]
                    meshes = Meshes(verts=verts, faces=faces, textures=textures)
                opti_joints_coco = opti_joints_coco.permute(0, 2, 1)

                # Render silhouettes
                focal_length = f.expand(-1, 2)
                cameras = PerspectiveCameras(in_ndc=False,
                                             focal_length=focal_length / downscale,
                                             principal_point=principal_point,
                                             image_size=torch.tensor([[out_h, out_w]]).repeat(batch_size, 1).to(device),
                                             R=torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(device),  # [bs, 3, 3]
                                             device=device
                                             )

                rendered_mask = renderer(meshes, cameras=cameras).flip([1, 2])  # [bs, out_h, out_w, 4]

                # Crop and resize to match seg_mask size (assuming seg_mask is [bs, 3, 224, 224])
                pred_mask = rendered_mask[:, ..., -1].clone()
                gt = seg_mask_down[:, 0].float().contiguous().clone()

                if batch_size > 1:
                    with torch.no_grad():
                        cur_mask = (data_batch['pad_mask'][:,0] - rendered_mask[...,-1]).clamp(0,1)
                    pred_mask = pred_mask + cur_mask

                if iteration < num_iterations/2:
                    pred = TF.gaussian_blur(pred_mask[:, None], kernel_size=17, sigma=8)
                    gt = TF.gaussian_blur(gt[:, None], kernel_size=17, sigma=8)
                    pred = F.interpolate(pred, scale_factor=0.25, mode="bilinear", align_corners=True)[:, 0]
                    gt = F.interpolate(gt, scale_factor=0.25, mode="bilinear", align_corners=True)[:, 0]
                else:
                    pred = TF.gaussian_blur(pred_mask[:, None], kernel_size=5, sigma=3)
                    gt = TF.gaussian_blur(gt[:, None], kernel_size=5, sigma=3)
                    pred = F.interpolate(pred, scale_factor=1., mode="bilinear", align_corners=True)[:, 0]
                    gt = F.interpolate(gt, scale_factor=1., mode="bilinear", align_corners=True)[:, 0]

                intersection = (pred * gt)
                total = (pred + gt)
                union = total - intersection

                iou_batched = (intersection + 1e-6).sum(-1).sum(-1) / (union + 1e-6).sum(-1).sum(-1)
                final_iou_batched = iou_batched.clone().detach()

                if iteration == 0:
                    best_betas = opti_betas.clone().detach()
                    best_body_pose = opti_body_pose.float().clone().detach()
                    if not self.convert_to_smpl:
                        best_left_hand_pose = opti_lhand_pose.clone().detach()
                        best_right_hand_pose = opti_rhand_pose.clone().detach()
                    best_global_orient = opti_global_orient.float().clone().detach()
                    best_transl = opti_transl.float().clone().detach()
                    best_transl_nopelvis = T.clone().detach()
                    best_f = f.clone().detach()
                    prev_best = iou_batched.clone().detach()

                for b_i in range(batch_size):
                    if not final_iou_batched[b_i].isnan() and final_iou_batched[b_i] > prev_best[b_i]:
                        best_betas[b_i] = opti_betas[b_i].clone().detach()
                        best_body_pose[b_i] = opti_body_pose[b_i].float().clone().detach()
                        if not self.convert_to_smpl:
                            best_left_hand_pose[b_i] = opti_lhand_pose[b_i].float().clone().detach()
                            best_right_hand_pose[b_i] = opti_rhand_pose[b_i].float().clone().detach()
                        best_global_orient[b_i] = opti_global_orient[b_i].float().clone().detach()
                        best_transl[b_i] = opti_transl[b_i].float().clone().detach()
                        best_transl_nopelvis[b_i] = T[b_i].clone().detach()
                        best_f[b_i] = f[b_i].clone().detach()
                        prev_best[b_i] = iou_batched[b_i].mean()

                iou_loss = 0.
                tz_loss = 0.
                txy_loss = 0.
                for b_i in range(batch_size):
                    iou_loss = iou_loss + (1. - iou_batched[b_i])
                    txy_loss = txy_loss + F.l1_loss(T[:, :2], torch.zeros_like(T[:, :2]))
                    tz_loss = tz_loss + F.l1_loss(T[:, -1:], input_T[:, -1:, 0].detach())
                iou_loss = iou_loss / batch_size
                tz_loss = tz_loss / batch_size
                txy_loss = txy_loss / batch_size


                opti_K = torch.eye(3)[None].repeat(batch_size, 1, 1).to(device)
                opti_K[:, 1, 1] = opti_K[:, 0, 0] = f[:, 0]
                opti_K[:, 0, 2] = out_w/2 * downscale; opti_K[:, 1, 2] = out_h/2 * downscale # undo downscaling for cx, cy

                opti_kpts_proj = opti_K @ (opti_joints_coco + T[:, :, None])
                opti_kpts_u = opti_kpts_proj[:, 0] / opti_kpts_proj[:, 2]
                opti_kpts_v = opti_kpts_proj[:, 1] / opti_kpts_proj[:, 2]
                opti_kpts_2d = torch.stack([opti_kpts_u, opti_kpts_v], dim=1)

                if self.convert_to_smpl:
                    loss = 1 * iou_loss + 0.1 * txy_loss + 0.1 * tz_loss
                    loss = loss + 8 * (good_flag[:, None] * 1. * F.l1_loss(opti_kpts_2d[:, :2], target_kpts_2d,
                                                                           reduction='none')
                                       / normalize_scale[:, None, None]).sum() / (good_flag.sum() * 2)  # \
                else:
                    loss = 1 * iou_loss + 0.1 * txy_loss + 0.1 * tz_loss

                    not_hand_good_flag = good_flag[:, None] & ~self.coco_wholebody_to_goliath_mapping_is_hand[None,None]
                    hand_good_flag = good_flag[:, None] & self.coco_wholebody_to_goliath_mapping_is_hand[None, None]
                    raw_kpt_loss = F.l1_loss(opti_kpts_2d[:, :2], target_kpts_2d, reduction='none')
                    if self.ignore_hand_kpts:
                        kpt_weight = not_hand_good_flag
                    else:
                        kpt_weight = (0.2 * not_hand_good_flag.sum() / hand_good_flag.sum()) * hand_good_flag + not_hand_good_flag
                    kpt_loss = 3 * 8 * (kpt_weight * raw_kpt_loss/ normalize_scale[:,None,None]).sum() / (good_flag.sum() * 2)  # \

                    # loss = loss + kpt_loss  # the keypoint loss seem to be more helpful than segmentation masks
                    loss = loss + kpt_loss * 0.5

                pose_reg   = (opti_body_pose      - orig_body_pose).pow(2).mean()
                orient_reg = (opti_global_orient  - orig_global_orient).pow(2).mean()
                # beta_reg   = (opti_betas          - orig_betas).pow(2).mean()
                loss = loss + 2 * pose_reg + orient_reg

                # Backpropagation
                loss.backward()
                optimizer.step()

                f.data.clamp_(min=10.0, max=1e4)
                Tz.data.clamp_(min=0.3, max=50)

        with torch.no_grad():

            if self.convert_to_smpl:
                best_output = self.body_model_train(
                    betas=best_betas,
                    body_pose=best_body_pose.float(),
                    global_orient=best_global_orient.float(),
                    transl=best_transl)
                best_verts = best_output['vertices']
                best_joints = best_output['joints']
                best_pelvis = best_joints[:, :1]
                best_transl_w_pelvis = best_pelvis[:, 0] + best_transl_nopelvis
                best_vert_nopelvis = best_verts - best_pelvis
                best_joints_nopelvis = best_joints - best_pelvis
            else:
                zero_like_transl = torch.zeros_like(best_transl)
                best_output = self.body_model_smplx_neutral(betas=best_betas,
                                              body_pose=best_body_pose,
                                              left_hand_pose=best_left_hand_pose,
                                              right_hand_pose=best_right_hand_pose,
                                              global_orient=best_global_orient,
                                              transl=best_transl,
                                              leye_pose=zero_like_transl,
                                              reye_pose=zero_like_transl,
                                              jaw_pose=zero_like_transl,
                                              expression=torch.zeros_like(best_betas).detach(),
                                              return_verts=True)
                best_verts = best_output.vertices
                best_joints_127 = F.pad(best_output.joints, (0, 0, 0, 17), mode='constant', value=0)
                best_joints_cocowholebody, _ = convert_kps(keypoints=best_joints_127, src='smplx', dst='coco_wholebody')
                best_joints = best_joints_cocowholebody[:, self.coco_wholebody_in_goliath]
                best_pelvis = best_joints_cocowholebody[:, 11:13].mean(1, keepdims=True)
                best_transl_w_pelvis = best_pelvis[:, 0] + best_transl_nopelvis
                best_vert_nopelvis = best_verts - best_pelvis
                best_joints_nopelvis = best_joints - best_pelvis

            if 'gt_transl_w_pelvis' in data_batch:
                # Print average loss and parameter values for the first element in the batch
                intersection = (pred * gt)
                total = (pred + gt)
                union = total - intersection

                # Compute IoU
                best_iou = (intersection + 1e-6).sum(-1).sum(-1) / (union + 1e-6).sum(-1).sum(-1)
                # for i in range(batch_size):
                #     plt.imshow((pred[i]-gt[i]).detach().cpu()); plt.colorbar(); plt.show()

                gt_transl_w_pelvis = data_batch['gt_transl_w_pelvis']
                gt_vert_nopelvis = data_batch['gt_vert_3d_no_pelvis']
                gt_joints_nopelvis = data_batch['gt_joints_3d_no_pelvis']
                gt_f = data_batch["posenet_K"][:, 0, 0]

                pve = ((gt_vert_nopelvis - best_vert_nopelvis) ** 2).sum(-1).sqrt().mean()

                best_joints_nopelvis = best_joints - best_pelvis
                mpjpe = ((gt_joints_nopelvis - best_joints_nopelvis) ** 2).sum(-1).sqrt().mean()

                best_T_error = (gt_transl_w_pelvis[:,0] - best_transl_w_pelvis).mean(0)

                orig_pelvis = predictions['pred_pelvis_3d'][:,0]
                orig_transl_w_pelvis = orig_pelvis + input_T[:,:,0]
                orig_T_error = (gt_transl_w_pelvis[:,0] - orig_transl_w_pelvis).mean(0)

                orig_f_percentage_error = ((input_f - gt_f).abs() / gt_f).mean() * 100
                best_f_percentage_error = ((best_f[:, 0]-gt_f).abs()/ gt_f).mean()*100

                print(f'Iter-{iteration}, '
                      f'f error: {orig_f_percentage_error:.3f} -> {best_f_percentage_error:.3f}%,'
                      f'IoU: {best_iou.mean()}, pve: {pve:.7f}, mpjpe: {mpjpe:.7f}\n'
                      f'T error: {orig_T_error.abs()} -> {best_T_error.abs()}')

        optimized_results = {'opti_betas': best_betas,
                             'opti_body_pose': best_body_pose.float(),
                             'opti_global_orient': best_global_orient.float(),
                             # NOTE: merge with T?
                             'opti_transl': best_transl,
                             'opti_vert_wpelvis': best_verts,
                             'opti_joints_wpelvis': best_joints,
                             'opti_pelvis': best_pelvis,
                             'opti_vert_nopelvis': best_vert_nopelvis,
                             'opti_joints_nopelvis': best_joints_nopelvis,
                             'opti_transl_w_pelvis': best_transl_w_pelvis,
                             'opti_transl_nopelvis': best_transl_nopelvis,
                             'opti_f': best_f,
                             'render_hw': torch.tensor((out_h, out_w), device=device)[None].expand(batch_size, -1)}
        if not self.convert_to_smpl:
            optimized_results.update({'opti_left_hand_pose': best_left_hand_pose,
                                      'opti_right_hand_pose': best_right_hand_pose,})

        predictions.update(optimized_results)
        has_seg_mask = torch.tensor(all_preds['has_seg_mask'], device=device) > 0

        return optimized_results

    def predict_pelvis_tz(self, data_batch, predictions):
        x = data_batch['depthnet_img_recrop']
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        # plt.imshow(x[0].permute(1, 2, 0).cpu()/3+0.6); plt.show()

        # extract intermediate feature from DepthAnythingV2
        if isinstance(self.depth_backbone, mmcv.parallel.distributed.MMDistributedDataParallel):
            depth_backbone = self.depth_backbone.module
        else:
            depth_backbone = self.depth_backbone
        features = depth_backbone.pretrained.get_intermediate_layers(x, depth_backbone.intermediate_layer_idx[
            depth_backbone.encoder],
                                                                     return_class_token=True)
        depth_pred, depth_feat = depth_backbone.depth_head(features, patch_h, patch_w)
        tmp_feat = resize(input=depth_feat[-1], size=(224, 224), mode="bilinear", align_corners=True)
        del depth_feat

        # predict pelvis depth Tz based on the intermediate features
        if self.use_depth_map:
            tmp_feat[:,-1:] = resize(input=depth_pred, size=(224, 224), mode="bilinear", align_corners=True)
        dino_features = self.depth_interface(tmp_feat.clone())
        del tmp_feat
        predictions_verts = self.depth_head(dino_features)
        predictions_verts['pred_z'] *= self.depth_scale


        predictions.update(predictions_verts)
        predictions['separate_pred_z'] = predictions_verts['pred_z']

    def predict_pose(self, data_batch, predictions):
        batch_size = data_batch['posenet_img'].shape[0]
        device = data_batch['posenet_img'].device

        # plt.imshow(data_batch['posenet_img_recrop'][0].permute(1, 2, 0).cpu() / 3 + 0.6); plt.show()

        aios_data_batch = {'img': data_batch['posenet_img_recrop'].clone(),
                           'img_shape': torch.tensor([data_batch['posenet_img_recrop'].shape[-2],
                                    data_batch['posenet_img_recrop'].shape[-1]], device=device)[None].repeat(batch_size, 1),
                           'body_bbox_center': [],
                           'body_bbox_size': [],
                           'pred_z': predictions['pred_z'].clone(),
                           'ann_idx': [torch.tensor([b_i], device=device) for b_i in range(batch_size)]
                           }

        # uses full image for pose estimator
        for b_i in range(batch_size):
            tmp = torch.tensor([0, 0, aios_data_batch['img'].shape[-1], aios_data_batch['img'].shape[-2]],
                               device=device)
            aios_data_batch['body_bbox_center'].append(tmp.clone())
            aios_data_batch['body_bbox_size'].append(tmp.clone())

        # predict pose
        aios_outputs, aios_targets, aios_data_batch_nc = self.pose_backbone(aios_data_batch)
        orig_target_sizes = torch.stack([t["size"] for t in aios_targets], dim=0)
        result, _ = self.pose_postprocessors['bbox'].forward_withgrad(aios_outputs, orig_target_sizes, aios_targets, aios_data_batch_nc)
        predictions['aios_outputs'] = aios_outputs

        # assumes that there's only one person in the image. gets the most confident prediction.
        # NOTE: one get get other predictions too, likely works
        all_expr, all_rhand_pose, all_lhand_pose, all_root_pose, all_pose, all_shape, all_pelvis \
            = [], [], [], [], [], [], []
        for b_i in range(batch_size):
            cur_result = result[b_i]
            all_pelvis.append(cur_result['smplx_kp3d'][0, 0, None])
            all_shape.append(cur_result['smplx_shape'][0])
            all_pose.append(cur_result['smplx_body_pose'][0])
            all_root_pose.append(cur_result['smplx_root_pose'][0])
            all_lhand_pose.append(cur_result['smplx_lhand_pose'][0])
            all_rhand_pose.append(cur_result['smplx_rhand_pose'][0])
            all_expr.append(cur_result['smplx_expr'][0])
        all_pelvis = torch.stack(all_pelvis, dim=0)
        all_shape = torch.stack(all_shape, dim=0)
        all_pose = torch.stack(all_pose, dim=0)
        all_root_pose = torch.stack(all_root_pose, dim=0)
        all_expr = torch.stack(all_expr, dim=0)
        all_lhand_pose = torch.stack(all_lhand_pose, dim=0)
        all_rhand_pose = torch.stack(all_rhand_pose, dim=0)

        predictions['pred_root_pose'] = all_root_pose
        predictions['pred_body_pose'] = all_pose
        predictions['pred_lhand_pose'] = all_lhand_pose
        predictions['pred_rhand_pose'] = all_rhand_pose
        predictions['pred_shape'] = all_shape

        # generate SMPL-X based on predicted parameters
        zero_transl = torch.zeros((batch_size, 3), device=device)
        pred_output = self.body_model_smplx_neutral(betas=all_shape,
                                      body_pose=all_pose,
                                      left_hand_pose=all_lhand_pose,
                                      right_hand_pose=all_rhand_pose,
                                      global_orient=all_root_pose,
                                      transl=zero_transl,
                                      leye_pose=torch.zeros_like(zero_transl),
                                      reye_pose=torch.zeros_like(zero_transl),
                                      jaw_pose=torch.zeros_like(zero_transl),
                                      expression=all_expr,
                                      return_verts=True)
        pred_verts = pred_output.vertices
        pred_pelvis = pred_output.joints[:, :1]
        pred_joints = pred_output.joints

        predictions['pred_pelvis_3d'] = pred_pelvis.clone()
        predictions['pred_vert_3d_w_pelvis'] = pred_verts.clone()
        predictions['pred_joints_3d_w_pelvis'] = pred_joints.clone()
        predictions['pred_vert_3d_no_pelvis'] = pred_verts - pred_pelvis
        predictions['pred_joints_3d_no_pelvis'] = pred_joints - pred_pelvis

    def convert_smplx_to_smpl(self, data_batch, predictions, all_preds):
        batch_size = data_batch['posenet_img'].shape[0]
        device = data_batch['posenet_img'].device

        # convert from SMPL-X to SMPL for evaluation
        smplx_mesh = {'vertices': predictions['pred_vert_3d_w_pelvis'],
                      'faces': self.smplx_faces[None].expand(batch_size, -1, -1).to(device)}
        convert_out = run_fitting(self.conversion_cfg, smplx_mesh,
                                  self.conversion_destination_model.to(device),
                                  self.conversion_def_matrix.to(device),
                                  self.conversion_mask_ids)
        pred_betas_smpl = convert_out['betas']
        pred_body_pose_smpl = matrix_to_axis_angle(convert_out['body_pose'])
        pred_global_orient_smpl = matrix_to_axis_angle(convert_out['global_orient'])
        pred_transl_smpl = convert_out['transl']
        pred_vert_smpl = convert_out['vertices']
        predictions['converted_smpl_betas'] = pred_betas_smpl
        predictions['converted_smpl_body_pose'] = pred_body_pose_smpl
        predictions['converted_smpl_global_orient'] = pred_global_orient_smpl
        predictions['converted_smpl_transl'] = pred_transl_smpl

        # re-generate SMPL mesh
        if isinstance(self.depth_backbone, mmcv.parallel.distributed.MMDistributedDataParallel):
            body_model = self.body_model_test.module
        else:
            body_model = self.body_model_test
        pred_joints_smpl = body_model.forward_joints(dict(vertices=pred_vert_smpl))[0]
        pred_pelvis = (pred_joints_smpl[:, self.right_hip_idx, :] + pred_joints_smpl[:, self.left_hip_idx, :])[:,None] / 2
        pred_joints_smpl_no_pelvis = pred_joints_smpl - pred_pelvis

        # save prediction for evaluation

        all_preds['vertices'] = predictions['converted_smpl_verts'] = pred_vert_smpl
        all_preds['keypoints_3d'] = pred_joints_smpl
        all_preds['smpl_beta'] = pred_betas_smpl
        all_preds['smpl_pose'] = torch.cat([pred_global_orient_smpl, pred_body_pose_smpl], dim=1)

        if 'gt_vert_3d_w_pelvis' in data_batch:
            gt_joints_smpl = body_model.forward_joints(dict(vertices=data_batch['gt_vert_3d_w_pelvis']))[0]
            gt_pelvis = (gt_joints_smpl[:, self.right_hip_idx, :] + gt_joints_smpl[:, self.left_hip_idx, :])[:,None] / 2
            gt_joints_smpl_no_pelvis = gt_joints_smpl - gt_pelvis

            mpjpe = ((pred_joints_smpl_no_pelvis - gt_joints_smpl_no_pelvis)**2).sum(dim=-1).sqrt().mean(dim=1).sum() / batch_size
            all_preds['mpjpe'] = mpjpe

            pred_vert_smpl_no_pelvis = pred_vert_smpl - pred_pelvis
            gt_vert_smpl_no_pelvis = data_batch['gt_vert_3d_w_pelvis'] - gt_pelvis
            pve = ((pred_vert_smpl_no_pelvis - gt_vert_smpl_no_pelvis)**2).sum(dim=-1).sqrt().mean(dim=1).sum() / batch_size
            all_preds['pve'] = pve

            print(f"MPJPE: {mpjpe:.6f}, PVE: {pve:.6f}")
            print(f"z_error: {(data_batch['smpl_transl'][:, 2] - predictions['pred_z'][:, 0]).abs().mean():.6f}")

    def crop_depth_images(self, all_preds, data_batch, scale_factor=1.25):
        """
        Crop depth images using the centroid of keypoints with a scaled bounding box.

        Args:
            keypoints: Tensor of shape [batch_size, 2, num_keypoints] containing x,y coordinates
            depth_images: Tensor of shape [batch_size, channels, height, width]
            scale_factor: Factor to scale the bounding box (default: 1.25)

        Returns:
            Cropped depth images tensor
        """
        keypoints = all_preds['ktps_list'][:,:,all_preds['good_flag'][0]]

        posenet_images = data_batch['posenet_img']

        batch_size = keypoints.shape[0]
        device = keypoints.device

        # Calculate bounding box extents
        min_coords, _ = torch.min(keypoints, dim=2)  # Shape: [batch_size, 2]
        max_coords, _ = torch.max(keypoints, dim=2)  # Shape: [batch_size, 2]

        # Calculate box size - use maximum of width and height for square box
        box_sizes = max_coords - min_coords  # Shape: [batch_size, 2]
        max_size, _ = torch.max(box_sizes, dim=1, keepdim=True)  # Shape: [batch_size, 1]
        box_sizes = torch.cat([max_size, max_size], dim=1)  # Make square

        # Calculate centroid for each sample in batch
        centroids = (max_coords + min_coords) / 2  # Shape: [batch_size, 2]

        # Scale box sizes
        scaled_sizes = box_sizes * scale_factor

        # Calculate crop boundaries
        half_sizes = scaled_sizes / 2
        top = centroids[:, 1] - half_sizes[:, 1]
        bottom = centroids[:, 1] + half_sizes[:, 1]
        left = centroids[:, 0] - half_sizes[:, 0]
        right = centroids[:, 0] + half_sizes[:, 0]

        # Normalize coordinates to [-1, 1] range for grid_sample
        height, width = posenet_images.shape[2:]

        boxes = torch.stack([
            2 * left / (width - 1) - 1,
            2 * top / (height - 1) - 1,
            2 * right / (width - 1) - 1,
            2 * bottom / (height - 1) - 1
        ], dim=1)  # Shape: [batch_size, 4]

        # Create sampling grid
        theta = torch.zeros(batch_size, 2, 3, device=device)
        theta[:, 0, 0] = (boxes[:, 2] - boxes[:, 0]) / 2
        theta[:, 0, 2] = (boxes[:, 2] + boxes[:, 0]) / 2
        theta[:, 1, 1] = (boxes[:, 3] - boxes[:, 1]) / 2
        theta[:, 1, 2] = (boxes[:, 3] + boxes[:, 1]) / 2

        # crop depth image with bbox
        grid = F.affine_grid(theta, (batch_size, 3, 518, 518), align_corners=True)
        data_batch['depthnet_img_recrop'] = F.grid_sample(posenet_images, grid, align_corners=True)

        # # visualize
        # for i in range(batch_size):
        #     u = keypoints[i, 0].detach().cpu()
        #     v = keypoints[i, 1].detach().cpu()
        #     cu = centroids[i, 0].detach().cpu()
        #     cv = centroids[i, 1].detach().cpu()
        #     plt.imshow(all_preds['seged_img'][i].permute(1, 2, 0).cpu()/255);
        #     plt.scatter(u, v, c='red', s=10, marker='o');
        #     plt.scatter(cu, cv, c='green', s=10, marker='o');
        #     plt.show()
        #     # plt.imshow(data_batch['depthnet_img_recrop'][i].permute(1, 2, 0).cpu() / 2 + 0.5);
        #     # plt.show()

    def crop_depth_images_using_segmask(self, all_preds, data_batch, scale_factor: float = 1.25, out_size: int = 518):
        """
        Square-crop depth / posenet images around the human mask.

        seg_masks  : all_preds['seg_masks']  (B, N, H, W)  0-1 / bool
        posenet_img: data_batch['posenet_img'] (B, C, H, W)

        Writes data_batch['depthnet_img_recrop']  (B, C, out_size, out_size)
        """
        seg_masks = all_preds['seg_masks']  # (B, N, H, W)
        posenet_images = data_batch['posenet_img']  # (B, C, H, W)
        B, _, H, W = posenet_images.shape
        device = posenet_images.device

        # 1. union over all people in each image
        union_masks = seg_masks.any(dim=1)  # (B, H, W)

        # 2. tight boxes (fallback: whole frame if mask empty)
        boxes = torch.zeros(B, 4, dtype=torch.float32, device=device)
        for b in range(B):
            if union_masks[b].any():
                boxes[b] = masks_to_boxes(union_masks[b].unsqueeze(0).float())[0]
            else:  # no person detected
                boxes[b] = torch.tensor([0., 0., W - 1., H - 1.], dtype=torch.float32, device=device)

        # 3. expand & force square
        centres = (boxes[:, :2] + boxes[:, 2:]) / 2  # (B, 2)
        wh = (boxes[:, 2:] - boxes[:, :2])  # (B, 2)
        wh *= scale_factor  # enlarge

        # make square: side = max(w, h)
        side = wh.max(dim=1, keepdim=True).values  # (B, 1)
        # side = side.clamp(max=min(H, W) - 1)  # can’t exceed frame
        half = side / 2

        new_min = centres - half
        new_max = centres + half

        # 4. shift square inside frame if it spills over (keeps it square)
        #   – left/right
        shift_x = (-new_min[:, 0]).clamp(min=0) - (new_max[:, 0] - (W - 1)).clamp(min=0)
        new_min[:, 0] += shift_x
        new_max[:, 0] += shift_x
        #   – top/bottom
        shift_y = (-new_min[:, 1]).clamp(min=0) - (new_max[:, 1] - (H - 1)).clamp(min=0)
        new_min[:, 1] += shift_y
        new_max[:, 1] += shift_y

        # 5. normalise to [-1, 1] for grid_sample
        left, top = new_min[:, 0], new_min[:, 1]
        right, bottom = new_max[:, 0], new_max[:, 1]

        boxes_norm = torch.stack([
            2 * left / (W - 1) - 1,
            2 * top / (H - 1) - 1,
            2 * right / (W - 1) - 1,
            2 * bottom / (H - 1) - 1
        ], dim=1)  # (B, 4)

        theta = torch.zeros(B, 2, 3, device=device)
        theta[:, 0, 0] = (boxes_norm[:, 2] - boxes_norm[:, 0]) / 2
        theta[:, 0, 2] = (boxes_norm[:, 2] + boxes_norm[:, 0]) / 2
        theta[:, 1, 1] = (boxes_norm[:, 3] - boxes_norm[:, 1]) / 2
        theta[:, 1, 2] = (boxes_norm[:, 3] + boxes_norm[:, 1]) / 2

        # 6. sample square crop
        grid = F.affine_grid(theta, (B, posenet_images.size(1), out_size, out_size), align_corners=True)
        data_batch['depthnet_img_recrop'] = F.grid_sample(posenet_images, grid, align_corners=True)



    def convert_coco_to_smplx(self, coco_keypoints):
        """
        Converts a COCO WholeBody keypoint array into a SMPL-X joint array using the provided mapping.

        Args:
            coco_keypoints (np.ndarray): Array of shape (N, 3) containing COCO keypoints.
            mapping (dict): Mapping from SMPL-X joint index to COCO keypoint index.
            num_smplx_joints (int): Number of SMPL-X joints expected.

        Returns:
            np.ndarray: Array of shape (num_smplx_joints, 3) with the converted joints.
        """
        # smplx_joints = np.zeros((127, 3), dtype=coco_keypoints.dtype)
        # for smplx_idx, coco_idx in self.smplx_to_cocowholebody.items():
        smplx_joints_cocowholebody = coco_keypoints[self.smplx_to_cocowholebody['cocowholebody']]
        return smplx_joints_cocowholebody


    def detect_segmentation_and_keypoints(self, data_batch, all_preds):
        batch_size = data_batch['posenet_img'].shape[0]
        device = data_batch['posenet_img'].device
        mask_h, mask_w = data_batch['posenet_img'].shape[-2:]

        # de-normalize for keypoint detector
        mp_input = data_batch['posenet_img'].clone()
        mp_input[:, 0] = mp_input[:, 0] * 58.395 + 123.675
        mp_input[:, 1] = mp_input[:, 1] * 57.120 + 116.280
        mp_input[:, 2] = mp_input[:, 2] * 57.375 + 103.53
        data_batch['mp_input'] = mp_input

        if 'smpl_betas' in data_batch:
            gt_mask_w_gt, gt_vertices = self.get_gt_seg_mask(data_batch)
            data_batch['gt_mask_w_gt_orig'] = gt_mask_w_gt.clone()
            if mask_h > mask_w:
                padding = (mask_h-mask_w)//2
                gt_mask_w_gt = gt_mask_w_gt[..., padding:padding + mask_w]
            elif mask_h < mask_w:
                padding = (mask_w - mask_h)//2
                gt_mask_w_gt = gt_mask_w_gt[:,:, padding:padding + mask_h]
        else:
            gt_mask_w_gt = None

        # use mediapipe to generate segmentation mask of the person
        # it expects image in 0-255
        if 'seg_masks' in data_batch:
            seg_masks = data_batch['seg_masks']
            orig_img = mp_input.clone()
            seged_img = mp_input.clone()
            seged_img[torch.logical_not(seg_masks.expand(-1,3,-1,-1).bool())] = 255
            has_seg_mask = [True] * batch_size
        else:
            seg_masks, seged_img, orig_img, has_seg_mask = self.detect_seg_mask(mp_input.to(torch.uint8), gt_mask_w_gt)
        # plt.imshow(seged_img[0].permute(1,2,0).cpu()/255); plt.show()

        # get bounding box using RTMPose, used for keypoint detection
        # rtm_input = seged_img.cpu().numpy() * 1.
        rtm_input = orig_img.cpu().numpy() * 1.
        rtm_input = rtm_input.transpose(0, 2, 3, 1)
        rtm_input_list = [rtm_input[i] for i in range(batch_size)]
        det_result = inference_detector(self.bbox_detector, rtm_input_list)
        bboxes_list = []
        for i in range(batch_size):
            pred_instance = det_result[i].pred_instances.cpu().numpy()
            bboxes = np.concatenate(
                (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
            bboxes = bboxes[np.logical_and(pred_instance.labels == self.kpt_args.det_cat_id,
                                           pred_instance.scores > self.kpt_args.bbox_thr)]
            bboxes = bboxes[nms(bboxes, self.kpt_args.nms_thr), :4]
            if len(bboxes) == 0:
                # NOTE: if no box, use mediapipe's box
                bboxes_list.append(np.array([[0, 0, rtm_input.shape[-3], rtm_input.shape[-2]]]))
            else:
                bboxes_list.append(bboxes[:1])

        bboxes_list = np.stack(bboxes_list, axis=0)

        # predict keypoints (mmpose) based on the bounding box
        pose_results = inference_topdown(self.kpt_estimator, rtm_input, bboxes_list)

        keypt_names = coco_kpt_names if self.convert_to_smpl else cocowholebody_kpt_names

        # get initial estimation
        ktps_list = []
        scores_list = []
        for i in range(batch_size):
            if has_seg_mask[i]:
                kpts = pose_results[i].pred_instances.keypoints[0]
                scores = pose_results[i].pred_instances.keypoint_scores[0]
                cur_coco_kpts = []
                cur_coco_scores = []
                if self.convert_to_smpl:
                    for coco_i in coco_kpt_names:
                        cores_idx = -1
                        for j_i in goliath_kpt_names:
                            if goliath_kpt_names[j_i] == coco_kpt_names[coco_i]['name']:
                                cores_idx = j_i
                        if cores_idx == -1:
                            cur_coco_kpts.append(np.zeros((2,)))
                            cur_coco_scores.append(np.zeros((1,)))
                        else:
                            cur_coco_kpts.append(kpts[cores_idx])
                            cur_coco_scores.append(scores[cores_idx])
                else:
                    for coco_i, goliath_i in coco_wholebody_to_goliath_mapping.items():
                        cur_coco_kpts.append(kpts[goliath_i])
                        cur_coco_scores.append(scores[goliath_i])
                cur_coco_kpts = np.stack(cur_coco_kpts, axis=0)
                ktps_list.append(cur_coco_kpts)
                scores_list.append(cur_coco_scores)
            else:
                if isinstance(self.depth_backbone, mmcv.parallel.distributed.MMDistributedDataParallel):
                    body_model = self.body_model_train.module
                else:
                    body_model = self.body_model_train
                cur_gt_kpt_2d, _ = convert_kps(data_batch['gt_joints_2d'][i:i+1], body_model.keypoint_src, 'coco', approximate=False)
                ktps_list.append(cur_gt_kpt_2d[0,:,:2].detach().cpu().numpy())
                cur_coco_scores = []
                for kpt_i in range(17):
                    if (0<cur_gt_kpt_2d[0,kpt_i,0]) * (cur_gt_kpt_2d[0,kpt_i,0] < mask_w) \
                        * (0<cur_gt_kpt_2d[0,kpt_i,1]) * (cur_gt_kpt_2d[0,kpt_i,1] < mask_h):
                        cur_coco_scores.append(1.)
                    else:
                        cur_coco_scores.append(0.)
                scores_list.append(cur_coco_scores)
                pass
        ktps_list = np.stack(ktps_list, axis=0)
        # ktps_list = ktps_list#[:,5:]  # leave out face kpts
        scores_list = np.stack(scores_list, axis=0)
        # scores_list = scores_list#[:, 5:]  # leave out face kpts
        # good_flag = ((scores_list > 0.05)
        good_flag = ((scores_list > 0.5)
                     & (ktps_list[...,0] >= 0) & (ktps_list[...,0] < mask_w)
                     & (ktps_list[...,1] >= 0) & (ktps_list[...,1] < mask_h))

        ktps_list = torch.tensor(ktps_list, device=device).permute(0,2,1)
        good_flag = torch.tensor(good_flag, device=device)

        all_preds['ktps_list'] = ktps_list
        all_preds['good_flag'] = good_flag
        all_preds['seg_masks'] = seg_masks
        all_preds['seged_img'] = seged_img
        all_preds['has_seg_mask'] = has_seg_mask
        # all_preds['bboxes_list'] = bboxes_list

        # for i in range(batch_size):
        #     plt.imshow(rtm_input_list[i] / 255);
        #     # plt.imshow(seged_img[i] / 255);
        #     # x_min, y_min, width, height = bboxes[0]
        #     # rect = plt.Rectangle((x_min, y_min), width, height, edgecolor='red', facecolor='none', linewidth=2)
        #     # plt.gca().add_patch(rect);
        #     plt.scatter(ktps_list[i,0].cpu(), ktps_list[i,1].cpu(), s=5)
        #     plt.show()




    def camera_solve(self, data_batch, predictions, all_preds):
        device = data_batch['posenet_img'].device
        batch_size, _, im_h, im_w = data_batch['posenet_img'].shape


        pred_transl = torch.cat([torch.zeros_like(predictions['pred_z'], device=device).expand(-1,2),
                                 predictions['pred_z']],dim=-1)[:,:,None]
        if self.convert_to_smpl:
            if isinstance(self.depth_backbone, mmcv.parallel.distributed.MMDistributedDataParallel):
                body_model = self.body_model_train.module
            else:
                body_model = self.body_model_train
            pred_joints_3d = body_model.forward_joints(dict(vertices=all_preds['vertices']))[0]
            pred_joints_3d, _ = convert_kps(
                pred_joints_3d,
                body_model.keypoint_src,
                'coco',
                approximate=False)
        else:
            pred_joints_3d = F.pad(predictions['pred_joints_3d_w_pelvis'], (0, 0, 0, 17), mode='constant', value=0)
            pred_joints_3d, conf_coco_full = convert_kps(keypoints=pred_joints_3d, src='smplx', dst='coco_wholebody')
            pred_joints_3d = pred_joints_3d[:, self.coco_wholebody_in_goliath]
        pred_joints_3d = pred_joints_3d.permute(0, 2, 1)

        init_focal = max(data_batch['posenet_img'].shape[-2], data_batch['posenet_img'].shape[-1])

        # init_proj_good_centered = []
        pred_f = []
        for b_i in range(batch_size):
            if self.convert_to_smpl:
                cur_good_flag = all_preds['good_flag'][b_i]
            else:
                cur_good_flag = all_preds['good_flag'][b_i] & ~self.coco_wholebody_to_goliath_mapping_is_hand
            cur_good_kpts2d = all_preds['ktps_list'][b_i, :, cur_good_flag]

            cur_good_kpts2d_mean = cur_good_kpts2d.mean(-1, keepdim=True)
            cur_good_kpts2d_centered = cur_good_kpts2d - cur_good_kpts2d_mean

            init_cx, init_cy, cur_K = im_w / 2, im_h / 2, torch.eye(3).to(device)
            cur_focal = init_focal
            for iter_i in range(3):
                cur_K[0, 0], cur_K[1, 1], cur_K[0, 2], cur_K[1, 2] = cur_focal, cur_focal, init_cx, init_cy
                coco_joints_homogeneous = (pred_joints_3d[b_i] + pred_transl[b_i])
                init_proj = cur_K @ coco_joints_homogeneous
                init_proj[0] /= init_proj[2]
                init_proj[1] /= init_proj[2]
                init_proj_good = init_proj[:, cur_good_flag]
                init_proj_mean = init_proj_good.mean(-1, keepdim=True)
                init_proj_good_centered = init_proj_good - init_proj_mean

                cur_f_scale = ((cur_good_kpts2d_centered ** 2).sum(0) / (init_proj_good_centered ** 2).sum(0)).sqrt().mean()
                cur_focal = cur_K[0, 0] = cur_K[1, 1] = cur_focal * cur_f_scale

                # lift detected joints to 3D
                smpl_joints_good = pred_joints_3d[b_i, :, cur_good_flag]
                center = torch.tensor([[init_cx], [init_cy]], device=device)
                xy_3d = (cur_good_kpts2d - center) * init_proj_good[-1:] / cur_focal
                txty = (xy_3d - smpl_joints_good[:2]).mean(-1, keepdim=True)
                pred_transl[b_i] = torch.cat([txty, predictions['pred_z'][b_i, None]],dim=0)
                # solved_txty.append(pred_transl)
            pred_f.append(cur_focal)

            # visualize
            # init_proj = cur_K @ (smpl_joints_good + pred_transl[b_i]).float()
            # init_proj[0] /= init_proj[2]
            # init_proj[1] /= init_proj[2]
            # pred_vert_smpl_transl = pred_vert_smpl[b_i].transpose(0,1) + pred_transl[b_i]
            # init_vert_proj = cur_K @ pred_vert_smpl_transl.float()
            # init_vert_proj[0] /= init_vert_proj[2]
            # init_vert_proj[1] /= init_vert_proj[2]
            # plt.imshow(data_batch['posenet_img'][b_i].cpu().permute(1,2,0)/3 + 0.5)
            # plt.scatter(init_vert_proj[0].cpu(), init_vert_proj[1].cpu(), c='orange', s=0.2)
            # plt.scatter(init_proj[0].cpu(), init_proj[1].cpu(), c='r')
            # plt.scatter(cur_good_kpts2d[0].detach().cpu(), cur_good_kpts2d[1].detach().cpu(), c='teal')
            # plt.xlim(-(im_h - im_w) // 2, im_w+(im_h - im_w)//2)
            # plt.ylim(im_h, 0)
            # plt.show()


        pred_f = torch.tensor(pred_f, device=device)
        pred_K = torch.eye(3).to(device)[None].repeat(batch_size, 1, 1)
        pred_K[:, 0, 0] = pred_K[:, 1, 1] = pred_f
        pred_K[:, 0, 2] = im_w / 2
        pred_K[:, 1, 2] = im_h / 2
        predictions['pred_K'] = pred_K
        all_preds['pred_transl'] = pred_transl
        all_preds['pred_f'] = pred_f

        optimized_results = self.solve_fTxTy_pytorch3d(all_preds, data_batch, predictions)

        return optimized_results


    def render_overlay(self, data_batch, predictions, optimized_results, all_preds):
        batch_size = data_batch['posenet_img'].shape[0]
        device = data_batch['posenet_img'].device
        pose_img_h, pose_img_w = data_batch['posenet_img'].shape[-2:]

        output_folder = self.temp_output_folder if self.temp_output_folder is not None \
            else f"{os.path.dirname(__file__)}/../../../results_vis"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        pose_img_size = torch.tensor(data_batch['posenet_img'].shape[-2:], device=device)

        if self.render_gt_instead:
            rendered_mesh_img = self.render_mesh(
                vertices=data_batch['gt_vert_3d_no_pelvis'],
                transl=data_batch['gt_transl_w_pelvis'],
                center=data_batch['center'].float(),
                scale=data_batch['scale'][:, 0].float(),
                focal_length_ndc=2 * data_batch['posenet_K'][:,0,0,None].repeat(1, 2).float() /
                                 torch.tensor(data_batch['posenet_img'].shape[-2:], device=device)[[1, 0]],
                px=data_batch['center'][:, 0],
                py=data_batch['center'][:, 1],
                img_wh=(pose_img_w, pose_img_h)
            )
        else:
            rendered_mesh_img = self.render_mesh(
                vertices=predictions['opti_vert_nopelvis'],
                transl=predictions['opti_transl_w_pelvis'],
                center=data_batch['center'].float(),
                scale=data_batch['scale'][:, 0].float(),
                focal_length_ndc=2 * predictions['opti_f'].repeat(1, 2).float() /
                                 torch.tensor(data_batch['posenet_img'].shape[-2:], device=device)[[1, 0]],
                px=data_batch['center'][:, 0],
                py=data_batch['center'][:, 1],
                img_wh=(pose_img_w, pose_img_h)
            )

        backbround_mask = rendered_mesh_img == 0
        foreground_mask = rendered_mesh_img != 0
        mp_input = all_preds['seged_img']
        blended_img = rendered_mesh_img.clone()
        blended_img[backbround_mask] = mp_input[backbround_mask].float() / 255
        blended_img[foreground_mask] = (blended_img[foreground_mask] * 0.9 + mp_input[foreground_mask] / 255 * 0.1)
        side_by_side = torch.cat([data_batch['mp_input'] / 255, blended_img], dim=-2)
        for plt_i in range(batch_size):
            plt.figure(figsize=(8, 8))
            plt.imshow(side_by_side[plt_i].permute(1, 2, 0).detach().cpu());
            good_flag = predictions['good_flag']
            plt.scatter(predictions['ktps_list'][plt_i, 0, good_flag[plt_i]].cpu(), predictions['ktps_list'][plt_i, 1, good_flag[plt_i]].cpu(), s=5)
            plt.scatter(predictions['ktps_list'][plt_i,0,good_flag[plt_i]].cpu(), predictions['ktps_list'][plt_i,1,good_flag[plt_i]].cpu() + pose_img_h, s=5)
            plt.title('overlayed');
            plt.axis('off')  # Turn off axis
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.show()

        f_ratio = predictions['opti_f'] / pose_img_size.max()

        # image_tensor = (blended_img * 255).clamp(0, 255).byte()
        image_tensor = (side_by_side * 255).clamp(0, 255).byte()
        transform_to_pil = transforms.ToPILImage()
        alpha_mask = (rendered_mesh_img[:, :1] > 0) * 1.
        png_img = torch.cat([rendered_mesh_img.clone(), alpha_mask], dim=1).permute(0, 2, 3, 1)
        png_img = ((png_img * 255).clamp(0, 255).byte()).cpu().numpy()
        fn_list = []
        for plt_i in range(batch_size):
            image = transform_to_pil(image_tensor[plt_i])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_folder}/{timestamp}_{plt_i}.jpg"
            image.save(filename)
            print(f"Image saved as {filename}\n\n\n")
            fn_list.append(filename)

            image = Image.fromarray(png_img[plt_i], 'RGBA')
            filename = filename.replace('.jpg', '_overlay.png')
            image.save(filename)

        #  render from side
        optimized_results['faces_list'] = self.smpl_faces
        optimized_results['scale'] = data_batch['scale'][:, 0].float()
        optimized_results['center'] = data_batch['center'].float()
        optimized_results['img_wh'] = (pose_img_w, pose_img_w)
        optimized_results['posenet_img'] = data_batch['posenet_img']
        torch.save(optimized_results, filename.replace('.jpg', '.pth'))


        return side_by_side

    def calculate_iou(self, data_batch, predictions, all_preds):
        pose_img_h, pose_img_w = data_batch['posenet_img'].shape[-2:]
        has_seg_mask = all_preds['has_seg_mask']
        batch_size = data_batch['posenet_img'].shape[0]
        # n_has_seg_mask = has_seg_mask.sum().item()
        n_has_seg_mask = sum(has_seg_mask)

        # vis & IoU
        pred_mask_w_gt = self.render_segmask(
            vertices=all_preds['vertices'],
            transl=all_preds['pred_transl'][:, :, 0],
            center=data_batch['center'].float(),
            scale=data_batch['scale'][:, 0].float(),
            focal_length_ndc=all_preds['pred_f'].float() / max(data_batch['posenet_img'].shape) * 2,
            px=data_batch['center'][:, 0],
            py=data_batch['center'][:, 1],
            img_res=max(pose_img_h, pose_img_w))

        opti_mask_w_gt = self.render_segmask(
            vertices=predictions['opti_vert_nopelvis'],
            transl=predictions['opti_transl_w_pelvis'],
            center=data_batch['center'].float(),
            scale=data_batch['scale'][:, 0].float(),
            focal_length_ndc=predictions['opti_f'][:, 0].float() / max(data_batch['posenet_img'].shape) * 2,
            px=data_batch['center'][:, 0],
            py=data_batch['center'][:, 1],
            img_res=max(pose_img_h, pose_img_w))

        if self.pmiou:
            gt_mask = data_batch['gt_mask_w_gt_orig']

            gt_mask_ = torch.zeros(batch_size, 24, gt_mask.shape[-1],
                                   gt_mask.shape[-1]).to(gt_mask.device)
            gt_mask_w_gt_ = torch.zeros(batch_size, 24, gt_mask.shape[-1],
                                        gt_mask.shape[-1]).to(gt_mask.device)
            for i in range(24):
                gt_mask_[:, i:i + 1] = (gt_mask == i + 1)
                gt_mask_w_gt_[:, i:i + 1] = (data_batch['gt_mask_w_gt_orig'] == i + 1)

            pred_mask = pred_mask_w_gt

            pred_mask_ = torch.zeros(batch_size, 24, pred_mask.shape[-1], pred_mask.shape[-1]).to(pred_mask.device)
            pred_mask_w_gt_ = torch.zeros(batch_size, 24, pred_mask.shape[-1], pred_mask.shape[-1]).to(pred_mask.device)
            for i in range(24):
                pred_mask_[:, i:i + 1] = (pred_mask == i + 1)
                pred_mask_w_gt_[:, i:i + 1] = (pred_mask_w_gt == i + 1)

            gt_mask_ = gt_mask_.view(batch_size, -1)
            pred_mask_ = pred_mask_.view(batch_size, -1)
            batch_u = (((gt_mask_ + pred_mask_) > 0) * 1.0).sum(1)
            batch_i = (gt_mask_ * pred_mask_).sum(1)
            batch_piou = batch_i / batch_u

            opti_mask = opti_mask_w_gt

            opti_mask_ = torch.zeros(batch_size, 24, pred_mask.shape[-1], pred_mask.shape[-1]).to(pred_mask.device)
            opti_mask_w_gt_ = torch.zeros(batch_size, 24, pred_mask.shape[-1], pred_mask.shape[-1]).to(pred_mask.device)
            for i in range(24):
                opti_mask_[:, i:i + 1] = (opti_mask == i + 1)
                opti_mask_w_gt_[:, i:i + 1] = (opti_mask_w_gt == i + 1)

            opti_mask_ = opti_mask_.view(batch_size, -1)
            opti_batch_u = (((gt_mask_ + opti_mask_) > 0) * 1.0).sum(1)
            opti_batch_i = (gt_mask_ * opti_mask_).sum(1)
            opti_batch_piou = opti_batch_i / opti_batch_u

            gt_mask_w_gt_ = gt_mask_w_gt_.view(batch_size, -1)
            opti_mask_w_gt_ = opti_mask_w_gt_.view(batch_size, -1)
            opti_batch_u_w_gt = (((gt_mask_w_gt_ + opti_mask_w_gt_) > 0) * 1.0).sum(1)
            opti_batch_i_w_gt = (gt_mask_w_gt_ * opti_mask_w_gt_).sum(1)
            opti_batch_piou_w_gt = opti_batch_i_w_gt / opti_batch_u_w_gt

        if self.miou:
            gt_mask = gt_mask.view(batch_size, -1)
            pred_mask = pred_mask.view(batch_size, -1)
            batch_u = (((gt_mask + pred_mask) > 0) * 1.0).sum(1)
            batch_i = ((gt_mask * pred_mask) > 0).sum(1)
            batch_iou = batch_i / batch_u

            opti_mask = opti_mask.view(batch_size, -1)
            opti_batch_u = (((gt_mask + opti_mask) > 0) * 1.0).sum(1)
            opti_batch_i = ((gt_mask * opti_mask) > 0).sum(1)
            opti_batch_iou = opti_batch_i / opti_batch_u

            gt_mask_w_gt_orig = data_batch['gt_mask_w_gt_orig'].view(batch_size, -1)
            opti_mask_w_gt = opti_mask_w_gt.view(batch_size, -1)
            opti_batch_u_w_gt = (((gt_mask_w_gt_orig + opti_mask_w_gt) > 0) * 1.0).sum(1)
            opti_batch_i_w_gt = ((gt_mask_w_gt_orig * opti_mask_w_gt) > 0).sum(1)
            opti_batch_iou_w_gt = opti_batch_i_w_gt / opti_batch_u_w_gt

        all_preds['raw_batch_miou'] = batch_iou * 100
        all_preds['raw_batch_pmiou'] = batch_piou * 100
        all_preds['opti_batch_miou'] = opti_batch_iou * 100
        all_preds['opti_batch_pmiou'] = opti_batch_piou * 100
        all_preds['opti_batch_miou_w_gt_mask'] = opti_batch_iou_w_gt * 100
        all_preds['opti_batch_pmiou_w_gt_mask'] = opti_batch_piou_w_gt * 100

        print(f'orig & opti IoU: {batch_iou.mean():.5f} -> {opti_batch_iou.mean():.5f}, part IoU: {batch_piou.mean():.5f} -> {opti_batch_piou.mean():.5f}')

        # if gt_mask_w_gt is not None:
        #     plt.imshow(gt_mask_w_gt[0, 0].detach().cpu()); plt.title(f'gt, z:{pred_transl[0,2,0]}'); plt.show()
        # plt.imshow(pred_mask_w_gt[0, 0].detach().cpu()); plt.title('kpt aligned'); plt.show()
        # plt.imshow(opti_mask_w_gt[0, 0].detach().cpu()); plt.title('optimized'); plt.show()

    def prepare_data(self, data_batch):
        batch_size = data_batch['posenet_img'].shape[0]

        if not self.is_demo:
            self.prepare_targets(data_batch)

        if not self.is_demo:
            data_batch['posenet_img_recrop'] = data_batch['posenet_img']
        else:
            # prepare posenet input
            posenet_images = data_batch['posenet_img']
            height, width = posenet_images.shape[2:]
            target_size = 400
            scale = min(target_size / height, target_size / width)
            new_h = int(height * scale)
            new_w = int(width * scale)
            resized = TF.resize(posenet_images, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR,
                                antialias=True)
            fill_tensor = torch.tensor((-2.1179, -2.0357, -1.8044), device=resized.device, dtype=resized.dtype).view(1,3,1,1)
            padded = fill_tensor.expand(batch_size, 3, target_size, target_size).clone()
            pad_h = (target_size - new_h) // 2
            pad_w = (target_size - new_w) // 2
            padded[:, :, pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
            data_batch['posenet_img_recrop'] = padded

        # remove background
        if 'seg_masks' in data_batch:
            seg_masks = torch.logical_not(data_batch['seg_masks'].expand(-1, 3, -1, -1).bool())
            for key in ['posenet_img']:
                data_batch[key][:,0][seg_masks[:,0]] = -2.1179
                data_batch[key][:,1][seg_masks[:,1]] = -2.0357
                data_batch[key][:,2][seg_masks[:,2]] = -1.8044

        return

    def forward_test(self, **data_batch):
        """
        """
        try:
            self.depth_backbone.eval()
            self.depth_interface.eval()
            self.depth_head.eval()
            self.pose_backbone.eval()

            batch_size = data_batch['img'].shape[0]
            device = data_batch['img'].device
            self.smpl_faces = self.smpl_faces.to(device)

            # Init
            if 'is_demo' in data_batch:
                self.is_demo = True
            else:
                self.is_demo = False

            predictions = dict()
            all_preds = {}
            all_preds['image_idx'] = data_batch['sample_idx']



            # Stage 1 ----------- Depth Estimation only
            if self.do_stage_1:
                self.prepare_targets_depthonly(data_batch)
                data_batch['depthnet_img_recrop'] = data_batch['depthnet_img']
                # depth backbone
                with torch.no_grad():
                    self.predict_pelvis_tz(data_batch, predictions)
                # pelvis_z_gt = data_batch['smpl_transl']
                pelvis_z_gt = data_batch['pelvis_camcoord']
                pelvis_z_pred = predictions['pred_z']
                z_error = (pelvis_z_gt[:, -1] - pelvis_z_pred[:, -1]).abs()
                all_preds['z_error'] = z_error
                all_preds['inv_z_error'] = (1./pelvis_z_gt[:, -1] - 1./pelvis_z_pred[:, -1]).abs()
                pred_output = dict2numpy(all_preds)
                return pred_output


            # Stage 2 ----------- Depth + Pose Estimation
            self.prepare_data(data_batch)


            # detect keypoint
            if not self.is_test_init_done:
                for name in self.freeze_modules:
                    for parameter in getattr(self, name).parameters():
                        parameter.requires_grad = False
                self.coco_wholebody_in_goliath = self.coco_wholebody_in_goliath.to(device)
                self.coco_wholebody_to_goliath_mapping_is_hand = self.coco_wholebody_to_goliath_mapping_is_hand.to(device)
                self.kpt_mask = self.kpt_mask.to(device)
                self.init_detectors(device)
            self.is_test_init_done = True

            self.detect_segmentation_and_keypoints(data_batch, all_preds)
            predictions['ktps_list'] = all_preds['ktps_list']
            predictions['good_flag'] = all_preds['good_flag']

            # prepare depth input
            if self.is_demo:
                self.crop_depth_images_using_segmask(all_preds, data_batch)
            else:
                data_batch['depthnet_img_recrop'] = data_batch['depthnet_img']

            # for b_i in range(batch_size):
            #     plt.imshow(data_batch['depthnet_img_recrop'][b_i].permute(1,2,0).cpu().numpy()/2.6+0.5); plt.show()



            # ------------------------------- predict Depth and SMPL-X ------------------------------------
            with torch.no_grad():
                self.predict_pelvis_tz(data_batch, predictions)
            self.predict_pose(data_batch, predictions)



            # ------------------------------- convert SMPL-X to SMPL ------------------------------------
            if self.convert_to_smpl:
                self.convert_smplx_to_smpl(data_batch, predictions, all_preds)
            else:
                all_preds['vertices'] = predictions['pred_vert_3d_w_pelvis']



            # ------------------------------------ SOLVE CAMERA -------------------------------------
            print("solve cameras")
            optimized_results = self.camera_solve(data_batch, predictions, all_preds)

            if self.convert_to_smpl:
                all_preds['smpl_beta'] = optimized_results['opti_betas']
                all_preds['smpl_pose'] = torch.cat([ optimized_results['opti_global_orient'],
                                                     optimized_results['opti_body_pose']], dim=1)
                all_preds['vertices'] = optimized_results['opti_vert_wpelvis']
                all_preds['keypoints'] = optimized_results['opti_joints_wpelvis']

            if self.render_and_save_imgs:
                print("rendering overlay images")
                side_by_side = self.render_overlay(data_batch, predictions, optimized_results, all_preds)

            if self.temp_output_folder is not None:
                save_data = {}
                for b_i in range(batch_size):
                    if self.id_list is not None:
                        cur_id = self.id_list[data_batch['id_idx'][b_i]]
                    else:
                        cur_id = data_batch['id_idx'][b_i]
                    save_data[cur_id] = {'smplx_betas': optimized_results['opti_betas'][b_i,None],
                                 'smplx_body_pose': optimized_results['opti_body_pose'][b_i,None],
                                 'smplx_global_orientation': optimized_results['opti_global_orient'][b_i,None],
                                 'smplx_translation': optimized_results['opti_transl'][b_i,None],
                                 'camera_translation': optimized_results['opti_transl_nopelvis'][b_i,None],
                                 'camera_focal_length': optimized_results['opti_f'][b_i,None],
                                 'camera_hw': optimized_results['render_hw'][b_i,None],
                                }
                    if not self.convert_to_smpl:
                        save_data[cur_id]['smplx_left_hand_pose'] = optimized_results['opti_left_hand_pose'][b_i, None]
                        save_data[cur_id]['smplx_right_hand_pose'] = optimized_results['opti_right_hand_pose'][b_i, None]
                    save_fn = os.path.join(self.temp_output_folder, f"{cur_id}.pth")
                    print(f'save results to {save_fn}')
                    torch.save(save_data[cur_id], save_fn)

            if self.is_demo:
                return {'dummy_return': data_batch['id_idx'][b_i]}
        except Exception as e:
            # Code to run if any error occurs
            # The variable 'e' holds the exception object
            print(f"An error occurred!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")

            traceback.print_exc()

            # Option 2 (string): capture it if you need to log/emit elsewhere
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            logging.error("Failure \n%s", tb_str)

            # IMPORTANT: preserve the original stack by using bare `raise`, not `raise e`
            # raise



        # ------------------------------------ Evaluation -------------------------------------
        if self.miou or self.pmiou:
            self.calculate_iou(data_batch, predictions, all_preds)
        else:
            print("not doing MIoU or p-MIoU")

        if 'smpl_transl' in data_batch:

            # focal error
            gt_f = data_batch['posenet_K'][:, 0, 0]
            pred_f = predictions['opti_f'][:, 0]
            all_preds['f_perc_error'] = (pred_f - gt_f).abs() / gt_f

            # Tz and 1/Tz error
            z_error = (data_batch['smpl_transl'][:, 2]
                       - predictions['separate_pred_z'][:, 0]).abs()
            all_preds['orig_z_error'] = z_error
            pelvis_z_gt = data_batch['gt_transl_w_pelvis'][:,0]
            pelvis_z_pred = predictions['opti_transl_w_pelvis']
            optim_z_error = (pelvis_z_gt[:, 2] - pelvis_z_pred[:, 2]).abs()
            all_preds['z_error'] = optim_z_error
            all_preds['inv_z_error'] = (1./pelvis_z_gt[:, 2] - 1./pelvis_z_pred[:, 2]).abs()
            all_preds['xy_error'] = ((pelvis_z_gt[:,:2] - pelvis_z_pred[:,:2])**2).sum(-1).sqrt()

            print(f"orig & opti IoU: {all_preds['raw_batch_miou'].mean():.5f} -> {all_preds['opti_batch_miou'].mean():.5f}, part IoU: {all_preds['raw_batch_pmiou'].mean():.5f} -> {all_preds['opti_batch_pmiou'].mean():.5f}")
            print(f'orig & opti z_error, {z_error.mean():.6f} -> {optim_z_error.mean():.6f}, xy_error: {all_preds["xy_error"].mean():.6f}, focal err%: {all_preds["f_perc_error"].mean():.6f}')

            # projection/distortion error
            standard_body_pose = torch.zeros((1, 69), device=device) * 0
            standard_betas = torch.zeros((1, 10), device=device) * 0
            standard_global_orient = torch.zeros((1, 3), device=device) * 0
            standard_output = self.body_model_train(
                betas=standard_betas,
                body_pose=standard_body_pose,
                global_orient=standard_global_orient)
            standard_vertices = standard_output['vertices'].expand(batch_size,-1,-1)

            # calculate distortion
            # NOTE: add gt_pelvis, which is the pelvis offset of the GT SMPL
            gt_verts_to_proj = standard_vertices.clone()
            gt_verts_to_proj[...,2:] = data_batch['smpl_transl'][:, None, 2:] # + gt_pelvis, which is the pelvis offset of the GT SMPL
            gt_perspective_proj = perspective_projection(gt_verts_to_proj, focal_length=1000)
            esti_verts_to_proj = standard_vertices.clone()
            esti_verts_to_proj[...,2:] += predictions['opti_transl_nopelvis'][:, None, -1:] # + gt_pelvis, which is the pelvis offset of the GT SMPL
            esti_perspective_proj = perspective_projection(esti_verts_to_proj, focal_length=1000)
            orthographic_proj = orthographic_projection(standard_vertices)
            scale = optimize_scale_and_translation(orthographic_proj, gt_perspective_proj)
            aligned_gt_persp_proj = gt_perspective_proj / scale
            aligned_esti_persp_proj = esti_perspective_proj / scale

            distortion = calculate_distortion_error(aligned_gt_persp_proj, aligned_esti_persp_proj)
            all_preds['distortion_error'] = distortion

        pred_output = dict2numpy(all_preds)

        return pred_output

    def forward_train(self, **kwargs):
        """Forward function for general training.

        For mesh estimation, we do not use this interface.
        """
        raise NotImplementedError('This interface should not be used in '
                                  'current training schedule. Please use '
                                  '`train_step` for training.')

    def compute_transl_loss(self, pred_transl, gt_transl, has_transl):
        batch_size = gt_transl.shape[0]
        conf = has_transl.float().view(batch_size, 1)
        loss = self.loss_transl_z(pred_transl, gt_transl, weight=conf)
        return loss

    def render_segmask(
        self,
        vertices: torch.Tensor,
        transl: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        focal_length_ndc: torch.Tensor,
        px: torch.Tensor,
        py: torch.Tensor,
        img_res: Optional[int] = 224,
    ):
        """Compute loss for part segmentations."""
        device = vertices.device
        seg_renderer = build_renderer(
            dict(type='segmentation', resolution=img_res, num_class=24))
        seg_renderer = seg_renderer.to(device)

        batch_size = vertices.shape[0]

        transl = transl.unsqueeze(1)
        K = torch.eye(3)[None].repeat_interleave(batch_size, 0)

        K[:, 0, 0] = focal_length_ndc * img_res / 2
        K[:, 1, 1] = focal_length_ndc * img_res / 2
        cx, cy = center.unbind(-1)

        K[:, 0, 2] = img_res / 2. - img_res * (cx - px) / scale
        K[:, 1, 2] = img_res / 2. - img_res * (cy - py) / scale
        cameras = build_cameras(
            dict(type='perspective',
                 image_size=(img_res, img_res),
                 in_ndc=False,
                 K=K,
                 convention='opencv')).to(device)

        mesh = Meshes(verts=vertices + transl,
                      faces=self.smpl_faces[None].repeat(
                          batch_size, 1, 1)).to(device)

        colors = torch.zeros_like(vertices)
        body_segger = body_segmentation('smpl')
        for i, k in enumerate(body_segger.keys()):
            colors[:, body_segger[k]] = i + 1
        mesh.textures = build_textures(
            dict(type='TexturesNearest', verts_features=colors))

        segmask = seg_renderer(mesh, cameras)
        segmask = segmask.permute(0, 3, 1, 2)
        return segmask

    def render_mesh(
        self,
        vertices: torch.Tensor,
        transl: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        focal_length_ndc: torch.Tensor,
        px: torch.Tensor,
        py: torch.Tensor,
        img_wh,
    ):
        """Compute loss for part segmentations."""
        device = vertices.device

        bs, n_pts = vertices.shape[:2]

        transl = transl.view(bs,1,3)
        K = torch.eye(3)[None].repeat_interleave(bs, 0)

        K[:, 0, 0] = focal_length_ndc[:, 0] * img_wh[0] / 2
        K[:, 1, 1] = focal_length_ndc[:, 1] * img_wh[1] / 2
        cx, cy = center.unbind(-1)

        K[:, 0, 2] = img_wh[0] / 2. - img_wh[0] * (cx - px) / scale
        K[:, 1, 2] = img_wh[1] / 2. - img_wh[1] * (cy - py) / scale
        cameras = build_cameras(
            dict(type='perspective',
                 image_size=(img_wh[1], img_wh[0]),
                 in_ndc=False,
                 K=K,
                 convention='opencv')).to(device)

        textures = torch.ones((bs, n_pts, 3), dtype=torch.float32, device=device)  # White color
        textures = TexturesVertex(verts_features=textures)  # [bs, n_pts, 3]
        mesh = Meshes(verts=vertices + transl, textures=textures,
                      faces=self.smpl_faces[None].repeat(bs, 1, 1).to(device)
                            if self.convert_to_smpl else
                            self.smplx_faces[None].repeat(bs, 1, 1).to(device))
        # Define PyTorch3D renderer with Rasterizer and Shader
        raster_settings = RasterizationSettings(
            image_size=(img_wh[1], img_wh[0]),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        directional_light = DirectionalLights(
            ambient_color=((.35, .35, .35),),  # Soft white color for directional light
            diffuse_color=((0.45, 0.45, 0.45),),
            specular_color=((0.5, 0.5, 0.5),),
            direction=((-0.25, -0.75, -2.5),),  # Direction pointing down and slightly towards the scene
            device=device
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=None,  # We'll define cameras per iteration
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=None,
                lights=directional_light,
                blend_params=BlendParams(background_color=(0.0, 0.0, 0.0))  # Black background
            )
        )  # [bs, 3]

        segmask = renderer(mesh, cameras=cameras)
        segmask = segmask.permute(0, 3, 1, 2)[:,:3]
        return segmask
