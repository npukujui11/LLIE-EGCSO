# Copyright (c) 2024
# DLN
# LBP (Lighten Back-Projection Module)
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True, isuseBN=False):
        super(ConvBlock, self).__init__()
        self.isuseBN = isuseBN
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        if self.isuseBN:
            self.bn = nn.BatchNorm2d(output_size)
        self.act = nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.isuseBN:
            out = self.bn(out)
        out = self.act(out)
        return out
class LightenBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(LightenBlock, self).__init__()
        codedim=output_size//2
        self.conv_Encoder = ConvBlock(input_size, codedim, 3, 1, 1, isuseBN=False)
        self.conv_Offset = ConvBlock(codedim, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Decoder = ConvBlock(codedim, output_size, 3, 1, 1,isuseBN=False)

    def forward(self, x):
        code= self.conv_Encoder(x)
        offset = self.conv_Offset(code)
        code_lighten = code+offset
        out = self.conv_Decoder(code_lighten)
        return out

class DarkenBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DarkenBlock, self).__init__()
        codedim=output_size//2
        self.conv_Encoder = ConvBlock(input_size, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Offset = ConvBlock(codedim, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Decoder = ConvBlock(codedim, output_size, 3, 1, 1,isuseBN=False)

    def forward(self, x):
        code= self.conv_Encoder(x)
        offset = self.conv_Offset(code)
        code_lighten = code-offset
        out = self.conv_Decoder(code_lighten)
        return out

class FusionLayer(nn.Module):
    def __init__(self, inchannel, outchannel, reduction=3):
        super(FusionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inchannel // reduction, inchannel, bias=False),
            nn.Sigmoid()
        )
        self.outlayer = ConvBlock(inchannel, outchannel, 1, 1, 0, bias=True)

    def forward(self, x):
        if len(x.shape)==3:
            x=x.unsqueeze(0)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        y = y + x
        y = self.outlayer(y)
        return y

class LBP(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(LBP, self).__init__()
        self.fusion = FusionLayer(input_size, output_size)
        self.conv1_1 = LightenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = DarkenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = LightenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1_1 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2_1 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.fusion(x)       #低光图像通过特征融合结构，相当于通过一个注意力
        hr = self.conv1_1(x)   #亮操作  得到  亮图像
        lr = self.conv2(hr)    #暗操作  得到  暗图像
        residue = self.local_weight1_1(x) - lr   #低光图像 减去 暗图像  得到 低暗残差图
        h_residue = self.conv3(residue)    #对低暗残差图 进行 亮操作
        hr_weight = self.local_weight2_1(hr)  #亮图像
        return hr_weight + h_residue

class DLN(nn.Module):
    def __init__(self, in_channels=3, dim=64):
        super(DLN, self).__init__()

        # brightness
        self.feat1 = ConvBlock(in_channels + 1, 2 * dim, 3, 1, 1)
        self.feat2 = ConvBlock(2 * dim, dim, 3, 1, 1)

        self.feat_out_1 = LBP(dim, dim, 3, 1, 1)
        self.feat_out_2 = LBP(2 * dim, dim, 3, 1, 1)
        self.feat_out_3 = LBP(3 * dim, dim, 3, 1, 1)

        self.feature = ConvBlock(4 * dim, dim, 3, 1, 1)
        self.out = nn.Conv2d(dim, in_channels, 3, 1, 1)

    def forward(self, x_ori):
        # data gate
        x = (x_ori - 0.5) * 2
        x_bright, _ = torch.max(x_ori, dim=1, keepdim=True)

        x_in = torch.cat([x, x_bright], dim=1)

        # feature extraction
        feature = self.feat1(x_in)
        feature_1_in = self.feat2(feature)
        feature_1_out = self.feat_out_1(feature_1_in)

        feature_2_in = torch.cat([feature_1_in, feature_1_out], dim=1)
        feature_2_out = self.feat_out_2(feature_2_in)

        feature_3_in = torch.cat([feature_1_in, feature_1_out, feature_2_out], dim=1)
        feature_3_out = self.feat_out_3(feature_3_in)

        feature_in = torch.cat([feature_1_in, feature_1_out, feature_2_out, feature_3_out], dim=1)
        feature_out = self.feature(feature_in)
        pred = self.out(feature_out) + x_ori

        return pred
