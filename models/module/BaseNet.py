# Copyright (c) 2024
# Licensed under the MIT License
# by @npukujui11
# BaseNet 是一个简单UNet网络
# 可用于对弱光图像进行一个简单的、初步的增强

from models.module.block import *

class BaseNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, norm=True):
        super(BaseNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels

        self.inc = DoubleConv(in_channels, 32, norm=norm)
        self.down1 = Down(32, 64, norm=norm)
        self.down2 = Down(64, 128, norm=norm)
        self.down3 = Down(128, 128, norm=norm)

        self.up1 = Up(256, 64, bilinear=True, norm=norm)
        self.up2 = Up(128, 32, bilinear=True, norm=norm)
        self.up3 = Up(64, 32, bilinear=True, norm=norm)
        self.outc = OutConv(32, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits
