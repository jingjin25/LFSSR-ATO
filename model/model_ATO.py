
import torch
import torch.nn as nn
import torch.nn.functional as functional
import math
from model.net_utils import *


class ATONet(nn.Module):
    def __init__(self, opt):
        super(ATONet, self).__init__()
        
        fn = opt.feature_num
        self.an = opt.angular_num
        self.an2 = self.an * self.an
        self.scale = opt.scale
        
        self.fea_conv0 = nn.Conv2d(1, fn, 3, 1, 1, bias=True)
        self.fea_resblock = make_layer(ResidualBlock, fn, opt.layer_num[0])
        
        self.pair_conv0 = nn.Conv2d(2*fn, fn, 3, 1, 1, bias=True)
        self.pair_resblock = make_layer(ResidualBlock, fn, opt.layer_num[1])
        self.pair_conv1 = nn.Conv2d(fn, fn, 3, 1, 1, bias=True)

        self.fusion_view_conv0 = nn.Conv2d(self.an2, fn, 3, 1, 1, bias=True)
        self.fusion_view_resblock = make_layer(ResidualBlock, fn, opt.layer_num[2])
        self.fusion_view_conv1 = nn.Conv2d(fn, 1, 3, 1, 1, bias=True)
        
        self.fusion_fea_conv0 = nn.Conv2d(fn, fn, 3, 1, 1, bias=True)
        self.fusion_fea_resblock = make_layer(ResidualBlock, fn, opt.layer_num[3])
       
        up = []
        for _ in range(int(math.log(self.scale,2))):
            up.append(nn.Conv2d(fn, 4*fn, 3, 1, 1, bias=True))
            up.append(nn.PixelShuffle(2))
            up.append(nn.ReLU(inplace=True))
        self.upsampler = nn.Sequential(*up)
        
        self.HRconv = nn.Conv2d(fn, fn//2, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(fn//2, 1, 3, 1, 1, bias=True)        

        self.relu = nn.ReLU(inplace=True)

    def forward(self, lf_lr, ref_ind):   
    
        N, an2, H, W = lf_lr.size()
        
        # reference view
        ref_view_lr = lf_lr[torch.arange(N), ref_ind].clone().view(N, 1, H, W)
        
        # individual LR feature extraction
        lf_fea = self.relu(self.fea_conv0(lf_lr.view(-1, 1, H, W)))
        lf_fea = self.fea_resblock(lf_fea).view(N, an2, -1, H, W)  # [N,an2,64,H,W]
        
        # pair feature
        lf_pair_fea = []
        ref_fea = lf_fea[torch.arange(N), ref_ind].clone()  # [N,64,H,W]
        for i in range(an2):
            aux_fea = lf_fea[torch.arange(N), i].clone()  # [N,64,H,W]
            pair_fea = torch.cat([ref_fea, aux_fea], 1)  # [N,128,H,W]
            lf_pair_fea.append(pair_fea)
        lf_pair_fea = torch.stack(lf_pair_fea, 1)  # [N,an2,128,H,W]

        # pair fusion
        lf_pair_fea = self.relu(self.pair_conv0(lf_pair_fea.view(N*an2, -1, H, W)))
        lf_pair_fea = self.pair_resblock(lf_pair_fea)
        lf_fea_aligned = self.pair_conv1(lf_pair_fea) # [N*an2,64,H,W]

        # all view fusion
        lf_fea_aligned = torch.transpose(lf_fea_aligned.view(N, an2, -1, H, W), 1, 2)  # [N,64,an2,H,W]
        ref_fea_fused = self.relu(self.fusion_view_conv0(lf_fea_aligned.view(-1, an2, H, W)))  # [N*64,64,h,w]
        ref_fea_fused = self.fusion_view_resblock(ref_fea_fused)  # [N*64,64,h,w]
        ref_fea_fused = self.relu(self.fusion_view_conv1(ref_fea_fused))  # [N*64,1,h,w]
        
        ref_fea_fused = self.relu(self.fusion_fea_conv0(ref_fea_fused.view(N, -1, H, W)))   #[N,64,h,w]
        ref_fea_fused = self.fusion_fea_resblock(ref_fea_fused)  # [N,64,h,w]

        # upsample
        ref_fea_hr = self.upsampler(ref_fea_fused)
        out = self.relu(self.HRconv(ref_fea_hr))
        out = self.conv_last(out)  # [N,1,H,W]

        # out
        base = functional.interpolate(ref_view_lr, scale_factor=self.scale, mode='bilinear', align_corners=False)
        out += base
        return out





    
 


 