# Copyright (c) 2024
# Licensed under the MIT License
# by @npkujui11
# 定义了不同模型的数据加载类，其中包含了三个数据集的定义
# `LowLightDataset` 用于加载低光数据集
# `LowLightEdgeDataset` 用于加使用边缘图像指导低光图像恢复的低光和边缘数据集
# `EdgeDataset` 用于加载边缘数据集

import os
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset


# 低光数据集定义
class LowLightDataset(Dataset):
    def __init__(self, low_dir, normal_dir, transform=None):
        self.low_dir = low_dir
        self.normal_dir = normal_dir
        self.transform = transform
        self.filenames = os.listdir(low_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        low_filename = self.filenames[idx]
        normal_filename = low_filename.replace("low", "normal")

        low_path = os.path.join(self.low_dir, low_filename)
        normal_path = os.path.join(self.normal_dir, normal_filename)

        low_image = Image.open(low_path)
        normal_image = Image.open(normal_path)

        if self.transform:
            low_image = self.transform(low_image)
            normal_image = self.transform(normal_image)

        return low_image, normal_image


class LowLightEdgeDataset(Dataset):
    def __init__(self, low_dir, normal_dir, edge_dir, transform=None):
        self.low_dir = low_dir
        self.normal_dir = normal_dir
        self.edge_dir = edge_dir
        self.transform = transform
        self.filenames = os.listdir(low_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        low_filename = self.filenames[idx]
        normal_filename = low_filename.replace("low", "normal")
        edge_filename = low_filename.replace("low", "edge")

        low_path = os.path.join(self.low_dir, low_filename)
        normal_path = os.path.join(self.normal_dir, normal_filename)
        edge_path = os.path.join(self.edge_dir, edge_filename)

        low_image = Image.open(low_path).convert('RGB')
        normal_image = Image.open(normal_path).convert('RGB')
        edge_image = Image.open(edge_path).convert('L')

        if self.transform:
            low_image = self.transform(low_image)
            normal_image = self.transform(normal_image)
            edge_image = self.transform(edge_image)

        return low_image, normal_image, edge_image, low_filename

import os
from torch.utils.data import Dataset
from PIL import Image

class LLEdgeDataset(Dataset):
    def __init__(self, low_dir, edge_dir, transform=None):
        self.low_dir = low_dir
        self.edge_dir = edge_dir
        self.transform = transform

        # **获取所有 `low image` 文件名（不含后缀）**
        self.low_filenames = {os.path.splitext(f)[0]: f for f in os.listdir(low_dir)}

        # **获取所有 `edge image` 文件，映射到不带后缀的文件名**
        self.edge_files = {os.path.splitext(f)[0]: os.path.join(edge_dir, f) for f in os.listdir(edge_dir)}

    def __len__(self):
        return len(self.low_filenames)

    def __getitem__(self, idx):
        low_filename_no_ext, low_filename = list(self.low_filenames.items())[idx]  # 取出文件名（无后缀）和完整文件名

        low_path = os.path.join(self.low_dir, low_filename)  # 获取完整路径

        # **确保 `edge` 目录中有对应的文件**
        if low_filename_no_ext in self.edge_files:
            edge_path = self.edge_files[low_filename_no_ext]
        else:
            print(f"⚠️ Warning: Edge file for {low_filename_no_ext} not found, skipping...")
            return None  # **跳过这个样本，防止 `NoneType` 错误**

        # **确保 `low_path` 存在**
        if not os.path.exists(low_path):
            print(f"⚠️ Warning: Low-light file {low_path} not found, skipping...")
            return None

        # 读取图像
        low_image = Image.open(low_path).convert('RGB')
        edge_image = Image.open(edge_path).convert('L')

        # 归一化
        if self.transform:
            low_image = self.transform(low_image)
            edge_image = self.transform(edge_image)

        return low_image, edge_image, os.path.basename(low_path)  # 返回完整的文件名


# 用于加载低光图像的类
class LLDataset(Dataset):
    def __init__(self, low_dir, transform=None):
        self.low_dir = low_dir
        self.transform = transform

        # **获取所有 `low image` 文件名（不含后缀）**
        self.low_filenames = {os.path.splitext(f)[0]: f for f in os.listdir(low_dir)}

    def __len__(self):
        return len(self.low_filenames)

    def __getitem__(self, idx):
        low_filename_no_ext, low_filename = list(self.low_filenames.items())[idx]  # 取出文件名（无后缀）和完整文件名

        low_path = os.path.join(self.low_dir, low_filename)  # 获取完整路径

        # **确保 `low_path` 存在**
        if not os.path.exists(low_path):
            print(f"⚠️ Warning: Low-light file {low_path} not found, skipping...")
            return None

        # 读取图像
        low_image = Image.open(low_path).convert('RGB')

        # 归一化
        if self.transform:
            low_image = self.transform(low_image)

        return low_image, os.path.basename(low_path)  # 返回完整的文件名


# 边缘数据集加载类的定义
"""
:param imgs_dir: 图像文件夹路径
:param edge_maps_dir: 边缘图像文件夹路径
:param transform: 数据预处理
"""
class EdgeDataset(Dataset):
    def __init__(self, imgs_dir, edge_maps_dir, transform=None):
        self.imgs_dir = imgs_dir
        self.edge_maps_dir = edge_maps_dir
        self.transform = transform
        self.filenames = os.listdir(imgs_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        imgs_filename = self.filenames[idx]
        normal_filename = imgs_filename.replace("imgs", "edge_maps").replace('.jpg', '.png')

        imgs_path = os.path.join(self.imgs_dir, imgs_filename)
        edge_maps_path = os.path.join(self.edge_maps_dir, normal_filename)

        imgs_image = Image.open(imgs_path)
        edge_maps_image = Image.open(edge_maps_path)

        if self.transform:
            imgs_image = self.transform(imgs_image)
            edge_maps_image = self.transform(edge_maps_image)

        return imgs_image, edge_maps_image, imgs_filename
