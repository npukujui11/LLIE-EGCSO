# Copyright (c) 2024
# Licensed under the MIT License
# by @npkujui11
# 用于检查边缘图片的尺寸是否与数据集图片相匹配

import os
from PIL import Image
import cv2

# 文件夹路径
low_folder = r"H:\datasets\Test-Dataset\ExDark\Low"
normal_folder = r"H:\datasets\Test-Dataset\ExDark\edge"

# 获取所有文件（忽略扩展名）
low_files = {os.path.splitext(f)[0]: f for f in os.listdir(low_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))}
normal_files = {os.path.splitext(f)[0]: f for f in os.listdir(normal_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))}

# 找到名称相同的图片（忽略扩展名）
common_names = set(low_files.keys()).intersection(set(normal_files.keys()))

# 记录尺寸不匹配的图片
mismatched_images = []

for name in common_names:
    low_path = os.path.join(low_folder, low_files[name])
    normal_path = os.path.join(normal_folder, normal_files[name])

    print(f"\n正在处理: {name} ({low_files[name]} vs {normal_files[name]})")

    try:
        # 尝试用 PIL 读取
        with Image.open(low_path) as low_img, Image.open(normal_path) as normal_img:
            low_size = low_img.size
            normal_size = normal_img.size
            print(f"✅ PIL 读取成功: {name} - Low尺寸: {low_size}, Normal尺寸: {normal_size}")

            if low_size != normal_size:
                print(f"⚠️ 尺寸不匹配: {low_size} vs {normal_size}")
                mismatched_images.append((name, low_files[name], normal_files[name], low_size, normal_size))

    except Exception as e:
        print(f"❌ PIL 读取失败: {name}，尝试使用 OpenCV ({e})")

        # 使用 OpenCV 读取
        low_img = cv2.imread(low_path)
        normal_img = cv2.imread(normal_path)

        if low_img is None or normal_img is None:
            print(f"❌ OpenCV 也无法读取 {name}，可能是损坏文件或格式错误")
            continue

        low_size = (low_img.shape[1], low_img.shape[0])  # (width, height)
        normal_size = (normal_img.shape[1], normal_img.shape[0])
        print(f"✅ OpenCV 读取成功: {name} - Low尺寸: {low_size}, Normal尺寸: {normal_size}")

        if low_size != normal_size:
            print(f"⚠️ 尺寸不匹配: {low_size} vs {normal_size}")
            mismatched_images.append((name, low_files[name], normal_files[name], low_size, normal_size))

# 输出结果
print("\n===============================")
if mismatched_images:
    print("❗ 以下图片的尺寸不匹配:")
    for name, low_file, normal_file, low_size, normal_size in mismatched_images:
        print(f"{name} ({low_file} vs {normal_file}): Low {low_size} vs Normal {normal_size}")
else:
    print("✅ 所有相同名称的图片尺寸匹配！")
