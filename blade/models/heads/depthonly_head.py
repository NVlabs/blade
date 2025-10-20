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


from torch import nn
from blade.models import BaseModule


class DiNOZNet(nn.Module):
    '''
    Z decoder
    '''

    def __init__(
        self,
        number_of_embed=1,  # 24 + 10
        embed_dim=256,
        nhead=4,
        dim_feedforward=1024,
        numlayers=2,
        max_depth=10,
    ) -> None:
        super(DiNOZNet, self).__init__()
        self.query = nn.Embedding(number_of_embed, embed_dim)
        self.embed_dim = embed_dim
        self.conv = nn.Conv2d(1, embed_dim, kernel_size=(1, 1))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=dim_feedforward,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=numlayers)
        self.z_out = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.max_depth = max_depth

    def forward(self, dino_features):
        bs =dino_features.size(0)  # B, 1, 56, 56
        decoder_out = self.decoder(
            self.query.weight.unsqueeze(0).repeat(bs, 1, 1),  # B, 1, 256
            dino_features.reshape(bs, self.embed_dim, -1).permute(0, 2,
                                                           1))  # B, 12544, 256
        # B, 1, 256
        z = self.z_out(decoder_out.reshape(bs, -1))  # B, 1
        z = self.sigmoid(z) * self.max_depth
        return {'pred_z': z}


class DepthOnlyHead(BaseModule):

    def __init__(
        self,
        feature_size=56,
        in_channels=256,
        znet_config=dict(
            number_of_embed=1,  # 24 + 10
            embed_dim=256,
            nhead=4,
            dim_feedforward=1024,
            numlayers=2,
        ),
        init_cfg=None,
    ):
        super(DepthOnlyHead, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.feature_size = feature_size

        self.dec_z = DiNOZNet(**znet_config)

    def forward(
            self,
            dino_feat
    ):
        dec_z_out = self.dec_z(dino_feat)
        pred_z = dec_z_out['pred_z']
        output = dict(pred_z=pred_z)
        return output
