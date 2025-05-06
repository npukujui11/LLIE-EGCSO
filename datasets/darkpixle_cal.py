# Copyright (c) 2024
# Licensed under the MIT License
# by @npkujui11
# 用于计算数据集中的暗区像素占比

import rawpy
import cv2
import os
import numpy as np


def convert_arw_to_png(arw_path, png_path):
    """将 ARW 文件转换为 PNG 文件"""
    with rawpy.imread(arw_path) as raw:
        rgb = raw.postprocess()  # 将 RAW 数据转换为 RGB
    cv2.imwrite(png_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))  # 保存为 PNG


def calculate_dark_pixel_percentage(image_path):
    """计算图像中灰度值小于 5 的像素占比"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
    total_pixels = image.size
    dark_pixels = np.sum(image < 5)  # 计算灰度值小于 5 的像素数量
    dark_pixel_percentage = (dark_pixels / total_pixels) * 100  # 计算占比
    return dark_pixel_percentage


def calculate_average_dark_pixel_percentage(folder_path):
    """计算文件夹中所有图像的平均暗区像素占比"""
    # 获取文件夹中所有 ARW 文件
    arw_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.arw')]

    # 初始化总暗区像素占比
    total_dark_percentage = 0

    # 遍历每张 ARW 文件
    for arw_file in arw_files:
        arw_path = os.path.join(folder_path, arw_file)
        png_path = os.path.join(folder_path, os.path.splitext(arw_file)[0] + '.png')

        # 将 ARW 转换为 PNG
        convert_arw_to_png(arw_path, png_path)

        # 计算暗区像素占比
        dark_percentage = calculate_dark_pixel_percentage(png_path)
        total_dark_percentage += dark_percentage

        # 打印单张图片的结果
        print(f"图片: {arw_file}, 暗区像素占比: {dark_percentage:.2f}%")

    # 计算平均暗区像素占比
    average_dark_percentage = total_dark_percentage / len(arw_files)

    return average_dark_percentage


# 文件夹路径
folder_path = 'G:\\CV-dataset\\Train\\SID\\Sony\\Sony\\Low'

# 计算平均暗区像素占比
average_dark_percentage = calculate_average_dark_pixel_percentage(folder_path)

print(f"平均暗区像素占比: {average_dark_percentage:.2f}%")