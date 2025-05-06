# Copyright (c) 2024
# Licensed under the MIT License
# by @npkujui11
# GT mean 操作

import os
import cv2
import numpy as np
from PIL import Image

# 假设im1和im2是PIL.Image对象
def gt_mean(im1, im2):
    # 获取im2的尺寸
    (h, w) = im2.size

    # 将im1调整为与im2相同的尺寸
    im1 = im1.resize((h, w))

    # 将PIL.Image对象转换为numpy数组
    im1 = np.array(im1)
    im2 = np.array(im2)

    # 计算im1和im2的灰度均值
    mean_restored = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY).mean()
    mean_target = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY).mean()

    # 调整im1的灰度均值
    im1 = np.clip(im1 * (mean_target / mean_restored), 0, 255)

    # 将numpy数组转换回PIL.Image对象
    im1 = Image.fromarray(im1.astype(np.uint8))

    return im1

# 定义处理文件夹的函数
"""
def process_folder(im1_folder, im2_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历im1_folder中的所有图片文件
    for filename in os.listdir(im1_folder):
        # 构建完整的文件路径
        im1_path = os.path.join(im1_folder, filename)
        im2_path = os.path.join(im2_folder, filename)

        # 检查im2_folder中是否存在对应的文件
        if not os.path.exists(im2_path):
            print(f"Warning: {filename} not found in {im2_folder}. Skipping.")
            continue

        # 打开图片
        im1 = Image.open(im1_path)
        im2 = Image.open(im2_path)

        # 调用gt_mean函数处理图片
        result = gt_mean(im1, im2)

        # 保存结果
        output_path = os.path.join(output_folder, filename)
        result.save(output_path)
        print(f"Processed and saved: {output_path}")
"""
# 定义处理文件夹的函数
def process_folder(im1_folder, im2_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历im1_folder中的所有图片文件
    for filename in os.listdir(im1_folder):
        # 构建完整的文件路径
        im1_path = os.path.join(im1_folder, filename)

        # 提取im1文件名中的数字部分，例如从"low00778_DSLR"提取"00778"
        base_filename = ''.join([c for c in filename if c.isdigit()])

        # 在im2_folder中查找包含相同数字部分的文件
        matched_files = [f for f in os.listdir(im2_folder) if base_filename in f]

        # 如果找到了匹配的文件
        if matched_files:
            im2_path = os.path.join(im2_folder, matched_files[0])

            # 打开图片
            im1 = Image.open(im1_path)
            im2 = Image.open(im2_path)

            # 调用gt_mean函数处理图片
            result = gt_mean(im1, im2)

            # 保存结果
            output_path = os.path.join(output_folder, filename)
            result.save(output_path)
            print(f"Processed and saved: {output_path}")
        else:
            print(f"Warning: No matching file found for {filename} in {im2_folder}. Skipping.")

"""
im1 = Image.open('G:\\模型复现结果\\√（2025）Our\\√Result_T_lolv2_syn_wde_wHVI_wLBP_wECA_wp_ep2000\\√VV\\P1010157.jpg')
im2 = Image.open('H:\\datasets\\Test-Dataset\\VV\\Normal\\P1010157.jpg')
result = gt_mean(im1, im2)
# 保存result
os.makedirs('./datasets/', exist_ok=True)  # 确保目录存在
result.save('../datasets/P1010157.jpg')
result.show()
"""
# 定义输入和输出文件夹路径

im1_folder = 'G:\\模型复现结果\\√（2025）Our\\√Result_T_lolv2_syn_wde_wHVI_wLBP_wECA_wp_ep2000\\√LSRW'
im2_folder = 'H:\\datasets\\LSRW\\Normal'
output_folder = 'G:\\模型复现结果\\√（2025）Our\\√Result_T_lolv2_syn_wde_wHVI_wLBP_wECA_wp_ep2000\\√LSRW_GT_mean'

# 处理文件夹中的所有图片
process_folder(im1_folder, im2_folder, output_folder)