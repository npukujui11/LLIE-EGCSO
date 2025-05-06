# Copyright (c) 2024
# Licensed under the MIT License
# by @npukujui11
# EDADE (Edge-adaptive Normalization, EDADE)
# EDADE original CLADE
# from
# https://github.com/tzt101/CLADE/blob/main/models/networks/normalization.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EDADE(nn.Module):
    def __init__(self, norm_nc):
        super(EDADE, self).__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # gamma/beta for two classes: 0=non-edge, 1=edge
        self.gamma = nn.Parameter(torch.Tensor(2, norm_nc))
        self.beta = nn.Parameter(torch.Tensor(2, norm_nc))

        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x, segmap):
        # x: (N, C, H, W), segmap: (N, 1, H, W)
        normalized = self.param_free_norm(x)
        mask = segmap.squeeze(1).long()  # (N, H, W)

        gamma = F.embedding(mask, self.gamma).permute(0, 3, 1, 2)  # (N, C, H, W)
        beta = F.embedding(mask, self.beta).permute(0, 3, 1, 2)

        return normalized * gamma + beta

class EDADEResnetBlock(nn.Module):
    def __init__(self, fin, fout):
        super(EDADEResnetBlock, self).__init__()
        fmiddle = min(fin, fout)

        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)

        self.norm_0 = EDADE(fin)
        self.norm_1 = EDADE(fmiddle)

        # optional learnable skip connection if in/out channels differ
        if fin != fout:
            self.skip_conv = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        else:
            self.skip_conv = None

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

    def forward(self, x, segmap):
        x_s = self.skip_conv(x) if self.skip_conv else x

        dx = self.conv_0(self.actvn(self.norm_0(x, segmap)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, segmap)))

        out = x_s + dx
        return out

if __name__ == "__main__":
    batch_size = 4
    input = torch.randn(batch_size, 4, 256, 256).cuda()
    print(f"input feature shape: {input.shape}")
    segmap = torch.randint(0, 2, (batch_size, 1, 256, 256), dtype=torch.long).cuda()
    print(f"segmap shape: {segmap.shape}")

    model = EDADEResnetBlock(4, 4).cuda()
    output = model(input, segmap)
    print(f"output shape: {output.shape}")
