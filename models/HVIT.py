# Copyright (c) 2024
# Licensed under the MIT License
# by @npukujui11
# 改进的HVI通道，详见论文

import torch
import torch.nn as nn
import torch.nn.functional as F

pi = 3.141592653589793


class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = nn.Parameter(torch.full([1], 0.2))  # k is reciprocal to the paper mentioned
        """
        self.gamma_G = nn.Parameter(torch.full([1], 0.2))  # Gamma for green channel
        self.gamma_B = nn.Parameter(torch.full([1], 0.2))  # Gamma for blue channel
        """
        self.gated = False
        self.gated2 = False
        self.alpha = 0.82

    def HVIT(self, img, edge):
        """
        :param
            img: 原始 RGB 图像，尺寸为 (B, 3, H, W)
            edge: 边缘图像，尺寸为 (B, 1, H, W)
        :return:
        """
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)

        # 将 RGB 空间的图片转换为 HUE
        hue[img[:, 2] == value] = 4.0 + ((img[:, 0] - img[:, 1]) / (value - img_min + eps))[img[:, 2] == value]
        hue[img[:, 1] == value] = 2.0 + ((img[:, 2] - img[:, 0]) / (value - img_min + eps))[img[:, 1] == value]
        hue[img[:, 0] == value] = (0.0 + ((img[:, 1] - img[:, 2]) / (value - img_min + eps))[img[:, 0] == value]) % 6
        hue[img.min(1)[0] == value] = 0.0
        hue = hue / 6.0

        """
        # Step1. 转换色调 HUE 通道
        hue = torch.where((hue >= 0) & (hue < 1 / 3), hue * 3 * self.gamma_G, hue)
        hue = torch.where((hue >= 1 / 3) & (hue < 2 / 3), 3 * (self.gamma_B - self.gamma_G) * (hue - 1 / 3) + self.gamma_G, hue)
        hue = torch.where((hue >= 2 / 3) & (hue <= 1), 3 * (1 - self.gamma_B) * (hue - 2 / 3) + 1, hue) % 6
        """

        # Step2. 饱和度调整
        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0

        # Step2. 转换为HSV空间
        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        # 根据图像的亮度（Value）通道自适应调整 k
        # k = torch.exp(-5 * (1 - value))
        E = edge[:, 0, :, :]
        E = torch.clamp(E, 0, 1)

        value_factor = value / (value.max() + eps)  # 基于亮度（Value通道）调整k
        edge_sensitive = torch.where(E == 1, torch.full_like(E, 0.5), torch.full_like(E, 0.2))  # 将边缘强度映射为色彩敏感度

        assert value_factor.shape[-2:] == edge_sensitive.shape[-2:], \
            "图像尺寸和边缘图像尺寸不匹配！"

        k = 0.8 * self.density_k + 0.10 * value_factor + 0.10 * edge_sensitive.unsqueeze(1)  # 小范围调整k值
        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        # color_sensitive = (torch.log(1 + value) + eps).pow(k)
        cx = (2.0 * pi * hue).cos()
        cy = (2.0 * pi * hue).sin()
        X = color_sensitive * saturation * cx
        Y = color_sensitive * saturation * cy
        Z = value
        xyz = torch.cat([X, Y, Z], dim=1)

        return xyz

    def PHVIT(self, img, edge):
        eps = 1e-8
        H, V, I = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]
        E = edge[:, 0, :, :]
        # clip
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        I = torch.clamp(I, 0, 1)
        E = torch.clamp(E, 0, 1)
        v = I

        """
        gamma_G, gamma_B = self.gamma_G, self.gamma_B
        """
        value_factor = v / (v.max() + eps)  # 基于亮度（Value通道）调整k
        edge_sensitive = torch.where(E == 1, torch.full_like(E, 0.5), torch.full_like(E, 0.2))  # 将边缘强度映射为色彩敏感度
        k = 0.8 * self.density_k + 0.10 * value_factor + 0.10 * edge_sensitive  # 小范围调整k值
        # 平滑 k 值变化，控制 color_sensitive 的范围
        # k = torch.clamp(k, 0.1, 0.5)

        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        # color_sensitive = (torch.log(1 + v) + eps).pow(k)
        H = H / (color_sensitive + eps)
        V = V / (color_sensitive + eps)

        # 色调
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)

        """
        H = torch.where((H >= 0) & (H < 1 / 3), H * 3 * gamma_G, H)
        H = torch.where((H >= 1 / 3) & (H < 2 / 3), 3 * (gamma_B - gamma_G) * (H - 1 / 3) + gamma_G, H)
        H = torch.where((H >= 2 / 3) & (H <= 1), 3 * (1 - gamma_B) * (H - 2 / 3) + 1, H) % 6
        """

        h = torch.atan2(V, H) / (2 * pi)
        h = h % 1
        s = torch.sqrt(H ** 2 + V ** 2 + eps)

        if self.gated:
            s = s * 1.3

        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb

if __name__ == "__main__":
    batch_size = 2
    img_height = 384
    img_width = 384

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input = torch.randn(batch_size, 3, img_height, img_width).to(device)
    edge = torch.randn(batch_size, 1, img_height, img_width).to(device)

    print(f"input shape: {input.shape}")
    print(f"edge shape: {edge.shape}")

    model = RGB_HVI().to(device)
    output = model.HVIT(input, edge)
    print(f"output shape: {output.shape}")
    output1 = model.PHVIT(output, edge)
    print(f"output1 shape: {output1.shape}")