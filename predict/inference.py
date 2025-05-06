# Copyright (c) 2024
# Licensed under the MIT License
# by @npukujui11
# 预测 EGCSO 模型

import os
import sys

# 获取当前脚本的目录（model 文件夹）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录（D:\program\LLIE\DEN）
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
# 添加到 sys.path
sys.path.append(parent_dir)

import time
import json
import torch
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from models.EGCSO_oEGEM import EGCSO
# from models.EGCSO import EGCSO
# from models.EGCSO_oEGEM_ICTwoEdgePrior import EGCSO
from datasets.dataloader import *
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from torchvision.utils import save_image
from torchvision.transforms import transforms, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# Image Preprocessing
transform = transforms.Compose([transforms.ToTensor()])
"""
dataset_paths = {
    'test_image': '../datasets/input/SICE/test',
    'test_edge': '../datasets/input/SICE/test-edge'
}
"""
dataset_paths = {
    'test_image': 'H:\\datasets\\LOL-v2\\Synthetic\\Test\\Low',
    'test_edge': 'H:\\datasets\\LOL-v2\\Synthetic\\Test\\edge'
}

def get_args():
    parser = argparse.ArgumentParser(description="Predict EGCSO model")
    parser.add_argument('--ckpt_path', type=str, default='../checkpoints/EGCSO/EGCSO_oEGEM_wde_wLBP_wECA_oP_ep2000.pth',
                        help='Path to checkpoint file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Path to output directory')
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    return parser.parse_args()


# 加载模型权重
def load_model(checkpoint_path):
    model = EGCSO().to(device)
    if os.path.exists(checkpoint_path):
        print(f"Restoring weights from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    return model


def load_dataset(paths, transform=None, batch_size=1, num_workers=1):
    test_dataset = LLEdgeDataset(
        paths['test_image'],
        paths['test_edge'],
        transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader


def evaluate_model(model, test_loader, args):
    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    pbar = tqdm(test_loader, desc="Evaluating Model")
    with torch.no_grad():
        for i, (img, edge, img_name) in enumerate(pbar):
            img, edge = img.to(device), edge.to(device)
            torch.cuda.empty_cache()
            output = model(img, edge)
            # **保存输出图像**
            output_filename = os.path.splitext(img_name[0])[0]
            output_ext = os.path.splitext(img_name[0])[1]
            output_path = os.path.join(args.output_dir, f"{output_filename}{output_ext}")
            save_image(output.clamp(0, 1), output_path)


# 主函数
def main():
    args = get_args()
    model = load_model(args.ckpt_path)
    test_loader = load_dataset(dataset_paths, transform)
    evaluate_model(model, test_loader, args)


if __name__ == '__main__':
    main()