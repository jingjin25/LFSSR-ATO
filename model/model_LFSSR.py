
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional
import numpy
import math

from model.net_utils import *


class LFSSRNet(nn.Module):
    def __init__(self, opt):
        super(LFSSRNet, self).__init__()

        # ATO model
        fn = opt.feature_num
        self.an = opt.angular_num
        self.an2 = self.an * self.an
        self.scale = opt.scale

        self.fea_conv0 = nn.Conv2d(1, fn, 3, 1, 1, bias=True)
        self.fea_resblock = make_layer(ResidualBlock, fn, opt.layer_num[0])

        self.pair_conv0 = nn.Conv2d(2 * fn, fn, 3, 1, 1, bias=True)
        self.pair_resblock = make_layer(ResidualBlock, fn, opt.layer_num[1])
        self.pair_conv1 = nn.Conv2d(fn, fn, 3, 1, 1, bias=True)

        self.fusion_view_conv0 = nn.Conv2d(self.an2, fn, 3, 1, 1, bias=True)
        self.fusion_view_resblock = make_layer(ResidualBlock, fn, opt.layer_num[2])
        self.fusion_view_conv1 = nn.Conv2d(fn, 1, 3, 1, 1, bias=True)

        self.fusion_fea_conv0 = nn.Conv2d(fn, fn, 3, 1, 1, bias=True)
        self.fusion_fea_resblock = make_layer(ResidualBlock, fn, opt.layer_num[3])

        up = []
        for _ in range(int(math.log(self.scale, 2))):
            up.append(nn.Conv2d(fn, 4 * fn, 3, 1, 1, bias=True))
            up.append(nn.PixelShuffle(2))
            up.append(nn.ReLU(inplace=True))
        self.upsampler = nn.Sequential(*up)

        self.HRconv = nn.Conv2d(fn, fn // 2, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(fn // 2, 1, 3, 1, 1, bias=True)

        # regularizer
        self.refine_conv0 = nn.Conv2d(1, 64, 3, 1, 1, bias=True)
        self.refine_sas = make_layer(AltFilter, opt.angular_num, opt.layer_num_refine)
        self.refine_conv1 = nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, lf_lr):

        N, an2, H, W = lf_lr.size()   ### N=1 ###

        # individual LR feature extraction
        lf_fea_lr = self.relu(self.fea_conv0(lf_lr.view(-1, 1, H, W)))
        lf_fea_lr = self.fea_resblock(lf_fea_lr)  # [N*an2,64,H,W]

        lf_inter = []
        group_len = 1   # reduce group_len when out of memory
        group_ind = torch.arange(0, an2, group_len)
        group_ind = torch.cat([group_ind, torch.tensor([an2])])

        for i in range(len(group_ind) - 1):
            cur_size = group_ind[i + 1] - group_ind[i]
            # construct pair
            ref_fea = lf_fea_lr[group_ind[i]:group_ind[i + 1]].view(cur_size, 1, -1, H, W).repeat(1, an2, 1, 1, 1)  # [group,-an2-,64,h,w]  ( [-an2-] means repeat dim )
            view_fea = lf_fea_lr.view(1, an2, -1, H, W).repeat(cur_size, 1, 1, 1, 1)  # [-group-,an2,64,h,w]

            pair_fea = torch.cat([ref_fea, view_fea], dim=2).view(cur_size * an2, -1, H, W)  # [group*an2,64*2,h,w]

            # pair feature
            pair_fea = self.relu(self.pair_conv0(pair_fea))
            pair_fea = self.pair_resblock(pair_fea)
            pair_fea = self.pair_conv1(pair_fea)  # [an*an2,64,H,W]
            pair_fea = pair_fea.view(cur_size, an2, -1, H, W).transpose(1, 2)  # [an,64,an2,H,W] ====> N=group

            # fusion
            fused_fea = self.relu(self.fusion_view_conv0(pair_fea.contiguous().view(-1, an2, H, W)))  # [N*64,an2,h,w]->[N*64,64,h,w]
            fused_fea = self.fusion_view_resblock(fused_fea)  # [N*64,64,h,w]
            fused_fea = self.relu(self.fusion_view_conv1(fused_fea))  # [N*64,1,h,w]

            fused_fea = self.relu(self.fusion_fea_conv0(fused_fea.view(cur_size, -1, H, W)))  # [N,64,h,w]
            fused_fea = self.fusion_fea_resblock(fused_fea)  # [N,64,h,w]
            ## upsample
            hr_fea = self.upsampler(fused_fea)
            hr_fea = self.relu(self.HRconv(hr_fea))
            res = self.conv_last(hr_fea)
            res = res.view(1, cur_size, self.scale * H, self.scale * W)  # [1,group,H,W]

            ## inter lf
            base = functional.interpolate(lf_lr[:, group_ind[i]:group_ind[i + 1]], scale_factor=self.scale,
                                          mode='bilinear', align_corners=False)
            # print(base.shape)
            lf_inter_group = res + base  # [1,group,2h,2w]
            lf_inter.append(lf_inter_group)

        lf_inter = torch.cat(lf_inter, 1)  # [1,an2,2h,2w]

        ### refine
        lf_out = self.relu(self.refine_conv0(lf_inter.view(N * an2, 1, self.scale * H, self.scale * W)))
        lf_out = self.refine_sas(lf_out)
        lf_out = self.refine_conv1(lf_out)
        lf_out = lf_out.view(N, an2, self.scale * H, self.scale * W)

        lf_out += lf_inter

        return lf_out





    
 


 