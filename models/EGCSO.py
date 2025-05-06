# Copyright (c) 2024
# Licensed under the MIT License
# by @npukujui11
# Edge-Guided and Color Space Optimization Network for LLIE
# EGCSO
import os
import sys

from torchvision.transforms import Resize

# 添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

from einops import rearrange
from models.HVIT import *
# from models.HVIT_ori import *
from models.module.transformer_utils import *
from models.LCA import *
from models.module.SPADE import *
# from models.module.EDADE import *
from thop import profile

class EGCSO(nn.Module):
    def __init__(self,
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False):
        super(EGCSO, self).__init__()

        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        # HV channels ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )

        # I channel ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
        )

        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)

        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)

        self.I_EdG4 = SPADEResnetBlock(ch4, ch4, 1)
        self.I_LCA4 = I_LCA(ch4, head4)

        self.I_EdG5 = SPADEResnetBlock(ch3, ch3, 1)
        self.I_LCA5 = I_LCA(ch3, head3)

        self.I_EdG6 = SPADEResnetBlock(ch2, ch2, 1)
        self.I_LCA6 = I_LCA(ch2, head2)

        self.trans = RGB_HVI().cuda()

    def forward(self, x, y):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x, y)
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)
        # low
        i_enc0 = self.IE_block0(i)                               # I 通道分支 特征提取模块
        i_enc1 = self.IE_block1(i_enc0)                          # I 通道分支 编码器01 下采样模块

        hv_0 = self.HVE_block0(hvi)                              # HV通道分支 特征提取模块
        hv_1 = self.HVE_block1(hv_0)                             # HV通道分支 编码器01 下采样模块

        i_jump0 = i_enc0                                         # I 通道分支 特征提取模块 跳跃连接
        hv_jump0 = hv_0                                          # HV通道分支 特征提取模块 跳跃连接

        i_enc2 = self.I_LCA1(i_enc1, hv_1)                       # I 通道分支 编码器01 交叉注意力机制
        hv_2 = self.HV_LCA1(hv_1, i_enc1)                        # HV通道分支 编码器01 交叉注意力机制

        v_jump1 = i_enc2                                         # I 通道分支 编码器01 跳跃连接
        hv_jump1 = hv_2                                          # HV通道分支 编码器01 跳跃连接

        i_enc2 = self.IE_block2(i_enc2)                          # I 通道分支 编码器02 下采样模块
        hv_2 = self.HVE_block2(hv_2)                             # HV通道分支 编码器02 下采样模块

        i_enc3 = self.I_LCA2(i_enc2, hv_2)                       # I 通道分支 编码器02 交叉注意力机制
        hv_3 = self.HV_LCA2(hv_2, i_enc2)                        # HV通道分支 编码器02 交叉注意力机制

        v_jump2 = i_enc3                                         # I 通道分支 编码器02 跳跃连接
        hv_jump2 = hv_3                                          # HV通道分支 编码器02 跳跃连接

        i_enc3 = self.IE_block3(i_enc2)                          # I 通道分支 编码器03 下采样模块
        hv_3 = self.HVE_block3(hv_2)                             # HV通道分支 编码器03 下采样模块

        i_enc4 = self.I_LCA3(i_enc3, hv_3)                       # I 通道分支 编码器03 交叉注意力机制
        hv_4 = self.HV_LCA3(hv_3, i_enc3)                        # I 通道分支 编码器03 交叉注意力机制

        # 将边缘指导图裁剪到与特征图相同的大小，插值模式为双线性插值，对齐方式为False
        #y_size4 = F.interpolate(y, size=(i_enc4.size(2), i_enc4.size(3)), mode='bilinear', align_corners=False)
        y_size4 = Resize(size=(i_enc4.size(2), i_enc4.size(3)))(y)
        i_enc4 = self.I_EdG4(i_enc4, y_size4)                    # I 通道分支 解码器04 编码引导模块

        i_dec4 = self.I_LCA4(i_enc4, hv_4)                       # I 通道分支 解码器04 交叉注意力机制
        hv_4 = self.HV_LCA4(hv_4, i_enc4)                        # HV通道分支 解码器04 交叉注意力机制

        hv_3 = self.HVD_block3(hv_4, hv_jump2)                   # HV通道分支 解码器04 上采样模块
        i_dec3 = self.ID_block3(i_dec4, v_jump2)                 # I 通道分支 解码器04 上采样模块

        # 将边缘指导图裁剪到与特征图相同的大小，插值模式为双线性插值，对齐方式为False
        #y_size5 = F.interpolate(y, size=(i_dec3.size(2), i_dec3.size(3)), mode='bilinear', align_corners=False)
        y_size5 = Resize(size=(i_dec3.size(2), i_dec3.size(3)))(y)
        i_dec3 = self.I_EdG5(i_dec3, y_size5)                    # I 通道分支 解码器05 编码引导模块

        i_dec2 = self.I_LCA5(i_dec3, hv_3)                       # I 通道分支 解码器05 交叉注意力机制
        hv_2 = self.HV_LCA5(hv_3, i_dec3)                        # HV通道分支 解码器05 交叉注意力机制

        hv_2 = self.HVD_block2(hv_2, hv_jump1)                   # HV通道分支 解码器05 上采样模块
        i_dec1 = self.ID_block2(i_dec2, v_jump1)                 # I 通道分支 解码器05 上采样模块

        #y_size6 = F.interpolate(y, size=(i_dec1.size(2), i_dec1.size(3)), mode='bilinear', align_corners=False)
        y_size6 = Resize(size=(i_dec1.size(2), i_dec1.size(3)))(y)
        i_dec1 = self.I_EdG6(i_dec1, y_size6)                    # I 通道分支 解码器06 编码引导模块

        i_dec0 = self.I_LCA6(i_dec1, hv_2)                       # I 通道分支 解码器06 交叉注意力机制
        hv_1 = self.HV_LCA6(hv_2, i_dec1)                        # HV通道分支 解码器06 交叉注意力机制

        i_dec0 = self.ID_block1(i_dec0, i_jump0)                 # I 通道分支 解码器06 上采样模块
        i_dec0 = self.ID_block0(i_dec0)                          # I 通道分支 特征整合模块
        hv_1 = self.HVD_block1(hv_1, hv_jump0)                   # HV通道分支 解码器06 上采样模块
        hv_0 = self.HVD_block0(hv_1)                             # HV通道分支 特征整合模块

        output = torch.cat([hv_0, i_dec0], dim=1)
        if output.shape[2:] != hvi.shape[2:]:
            output = F.interpolate(output, size=(hvi.shape[2], hvi.shape[3]), mode='bilinear', align_corners=False)
        output_hvi = output + hvi  # 残差连接
        output_rgb = self.trans.PHVIT(output_hvi, y)             # 将HVI通道转换为RGB通道

        return output_rgb

    def HVIT(self, x, y):
        hvi = self.trans.HVIT(x, y)
        return hvi


if __name__ == "__main__":
    batch_size = 2
    img_height = 768
    img_width = 768

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input = torch.randn(batch_size, 3, img_height, img_width).to(device)
    edge = torch.randn(batch_size, 1, img_height, img_width).to(device)
    print(f"input shape: {input.shape}")
    print(f"edge shape: {edge.shape}")

    model = EGCSO().to(device)
    output = model(input, edge)
    print(f"output shape: {output.shape}")

    flops, params = profile(model, inputs=(input, edge), verbose=False)
    print(f"EGCSO FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"EGCSO Params: {params / 1e6:.2f} M")


