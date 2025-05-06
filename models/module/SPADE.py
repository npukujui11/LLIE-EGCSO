# Copyright (c) 2024
# Licensed under the MIT License
# SPADE (Spatially Adaptive Normalization)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, nhidden=128, kernel_size=3):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        padding = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=padding)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=padding)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        act = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(act)
        beta = self.mlp_beta(act)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, semantic_nc):
        super().__init__()

        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)

        # define normalization layers
        self.norm_0 = SPADE(fin, semantic_nc, kernel_size=5)
        self.norm_1 = SPADE(fmiddle, semantic_nc, kernel_size=5)

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

    def forward(self, x, seg):
        # Part 1. pass the input through a ResNet block
        x_s = x

        # Part 2. apply normalization
        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx
        return out

if __name__ == "__main__":
    batch_size = 4
    input = torch.randn(batch_size, 4, 256, 256).cuda()
    print(f"input feature shape: {input.shape}")
    segmap = torch.randn(batch_size, 1, 256, 256).cuda()
    print(f"segmap shape: {segmap.shape}")

    model = SPADEResnetBlock(4, 4, 1).cuda()
    output = model(input, segmap)
    print(f"output shape: {output.shape}")