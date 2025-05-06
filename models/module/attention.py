# Copyright (c) 2024
# Licensed under the MIT License
# by @npukujui11
# 模型中会使用到的各种注意力机制模块

import torch
import torch.nn as nn
import torch.nn.functional as F

# CBAM块中的通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


# CBAM块中的空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)


# 定义轴向深度卷积模块
class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x


# Attention Residual Multi-scale Block
# ARMB模块，结合通道注意力机制和空间注意力机制的一种多尺度残差注意力模块
class ARMB(nn.Module):
    def __init__(self, in_channels):
        super(ARMB, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.axial_dw = AxialDW(in_channels, mixer_kernel=(3, 3))
        self.pw = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # First branch
        x1_down = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x1_sa = self.spatial_attention(x1_down)
        x1_conv = self.conv3x3(x1_sa)
        x1_up = F.interpolate(x1_conv, size=x.size()[2:], mode='bilinear', align_corners=True)

        # Second branch
        x2_down = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        x2_sa = self.spatial_attention(x2_down)
        x2_conv = self.conv3x3(x2_sa)
        x2_up = F.interpolate(x2_conv, size=x.size()[2:], mode='bilinear', align_corners=True)

        # Concatenate and 1x1 Conv
        concat = torch.cat([x1_up, x2_up], dim=1)
        x3 = self.pw(concat)

        # Channel Attention
        x4 = self.channel_attention(x3) * x3

        # Skip connection
        out = x + x4

        return out
