# Copyright (c) 2024
# Licensed under the MIT License
# by @npukujui11
# LCA模块的代码

import torch
import torch.nn as nn
from einops import rearrange
from models.module.transformer_utils import *
from models.module.LBP import *
from models.module.cbam import *

# Cross Attention Block
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# Intensity Enhancement Layer
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x

class IEL_mod(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False, reduction=8, sa_kernel_size=7):
        super(IEL_mod, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        # Input projection
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # Depth-wise convolution branches
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)

        # Spatial Attention
        self.spatial_attention1 = SpatialAttention(kernel_size=sa_kernel_size)
        self.spatial_attention2 = SpatialAttention(kernel_size=sa_kernel_size)
        self.sa_conv1x1 = nn.Conv2d(1, hidden_features, kernel_size=1, bias=bias)

        # Channel Attention
        self.channel_attention = ChannelAttention(hidden_features)

        # Output projection
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.Tanh = nn.ReLU()

    def forward(self, x):
        # Input projection
        x = self.project_in(x)

        # Split into two branches
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x1 = self.spatial_attention1(x1)
        x1 = self.sa_conv1x1(x1)

        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x2 = self.spatial_attention2(x2)
        x2 = self.sa_conv1x1(x2)

        x = x1 * x2
        x = self.channel_attention(x)
        # Channel Attention
        x = self.project_out(x)

        return x

class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.ReLU()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1).permute(0, 2, 1)  # (B, 1, C)
        y = self.conv(y).permute(0, 2, 1)  # (B, C, 1)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


# Lightweight Cross Attention
class HV_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(HV_LCA, self).__init__()
        self.gdfn1 = IEL(dim)  # IEL and CDL have same structure
        self.gdfn2 = ECA(dim)
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn2(self.gdfn1(self.norm(x)))
        return x


class I_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        # self.gdfn = IEL_mod(dim)
        # self.gdfn = ECALayer(dim)
        self.gdfn = LBP(dim, dim, 3, 1, 1)
        self.ffn = CAB(dim, num_heads, bias=bias)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = x + self.gdfn(self.norm(x))
        return x
