# Copyright (c) 2024
# Licensed under the MIT License
# by @npkujui11
# 用于将文件名中的 'normal' 替换为 'low'

import os

def rename_files(directory):
    """
    将目录中的文件名从 'normalxxxxxx.png' 修改为 'lowxxxxxx.png'
    :param directory: 文件所在的文件夹路径
    """
    for filename in os.listdir(directory):
        if filename.startswith("normal") and filename.endswith(".png"):
            new_filename = filename.replace("normal", "low", 1)  # 只替换前缀
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)

            # 确保不会覆盖已有文件
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"✅ Renamed: {filename} → {new_filename}")
            else:
                print(f"⚠️ Skipped: {new_filename} already exists")

# 设置目标文件夹路径
target_directory = "H:\\datasets\\LOL-v2\\Real_captured\\Test\\edge"  # 替换为你的实际路径

rename_files(target_directory)
