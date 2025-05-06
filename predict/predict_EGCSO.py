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
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
from models.EGCSO import EGCSO
from datasets.dataloader import *
from torch.utils.data import DataLoader
from models.ops.iqa import calculate_lpips, calculate_nima, calculate_loe
from pytorch_msssim import ssim
from torchvision.utils import save_image
from torchvision.transforms import transforms, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Preprocessing
transform = transforms.Compose([transforms.ToTensor()])


dataset_paths = {
    'test_image': 'H:\\datasets\\Test-Dataset\\DARK FACE\\Normal',
    'test_normal': 'H:\\datasets\\Test-Dataset\\DARK FACE\\Low',
    'test_edge': 'H:\\datasets\\Test-Dataset\\DARK FACE\\edge'
}

def get_args():
    parser = argparse.ArgumentParser(description="Predict EGCSO model")
    parser.add_argument('--ckpt_path', type=str, default='../checkpoints/EGCSO/EGCSO_wde_wLBP_wECA_oP_ep2000.pth',
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
    test_dataset = LowLightEdgeDataset(
        paths['test_image'],
        paths['test_normal'],
        paths['test_edge'],
        transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader


def predict(model, img_path, edge_path, transform=None):
    img = Image.open(img_path).convert('RGB')
    edge = Image.open(edge_path).convert('L')
    if transform:
        img = transform(img).unsqueeze(0).to(device)  # 添加batch维度并转移到设备
        edge = transform(edge).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img, edge)
    return output.squeeze(0).cpu()  # 移除batch维度并返回到CPU


def evaluate_model(model, test_loader, args):
    model.eval()
    # total_ssim, total_psnr, total_lpips, total_nima, total_loe = 0, 0, 0, 0, 0

    os.makedirs(args.output_dir, exist_ok=True)

    pbar = tqdm(test_loader, desc="Evaluating Model")
    with torch.no_grad():
        for i, (img, normal, edge, img_name) in enumerate(pbar):
            img, normal, edge = img.to(device), normal.to(device), edge.to(device)

            """
            ori_h, ori_w = img.shape[2], img.shape[3]
            img_resize = F.interpolate(img, size=(384, 384), mode='bilinear', align_corners=True)
            edge_resize = F.interpolate(edge, size=(384, 384), mode='bilinear', align_corners=True)
            
            outputs = model(img_resize, edge_resize)

            outputs = F.interpolate(outputs, size=(ori_h, ori_w), mode='bilinear', align_corners=True)
            """
            output = model(img, edge)
            # **保存输出图像**
            output_filename = os.path.splitext(img_name[0])[0]
            output_ext = os.path.splitext(img_name[0])[1]
            output_path = os.path.join(args.output_dir, f"{output_filename}{output_ext}")
            save_image(output.clamp(0, 1), output_path)

            """
            # SSIM
            ssim_value = ssim(outputs, normal, data_range=1.0, size_average=True)
            total_ssim += ssim_value.item()

            # PSNR
            mse = F.mse_loss(outputs, normal)
            psnr = 10 * torch.log10(1 / mse)
            total_psnr += psnr.item()

            # LPIPS
            lpips = calculate_lpips(outputs, normal)
            total_lpips += lpips.item()

            # NIMA
            nima = calculate_nima(outputs)
            total_nima += nima.item()

            # LOE
            loe = calculate_loe(outputs, normal)
            total_loe += loe.item()
            """
        """
        avg_ssim = total_ssim / len(test_loader)
        avg_psnr = total_psnr / len(test_loader)
        avg_lpips = total_lpips / len(test_loader)
        avg_nima = total_nima / len(test_loader)
        avg_loe = total_loe / len(test_loader)
        """

    # return avg_ssim, avg_psnr, avg_lpips, avg_nima, avg_loe

# 主函数
def main():
    args = get_args()
    model = load_model(args.ckpt_path)

    # 定义图像和边缘图像路径
    image_paths = [
        'H:\\datasets\\Test-Dataset\\DARK FACE\\Low\\0.png',
        'H:\\datasets\\Test-Dataset\\DARK FACE\\Low\\5.png',
        'H:\\datasets\\Test-Dataset\\DARK FACE\\Low\\10.png',
        'H:\\datasets\\Test-Dataset\\DARK FACE\\Low\\15.png'
    ]
    """
    image_paths = [
        'G:\\CV-dataset\\测试数据集\\VV\\P1010062.jpg',
        'G:\\CV-dataset\\测试数据集\\NPE\\birds.jpg',
        'G:\\CV-dataset\\测试数据集\\MEF\\Tower.png',
        'G:\\CV-dataset\\测试数据集\\LIME\\7.bmp'
    ]
    """
    edge_paths = [
        'H:\\datasets\\Test-Dataset\\DARK FACE\\edge\\0.png',
        'H:\\datasets\\Test-Dataset\\DARK FACE\\edge\\5.png',
        'H:\\datasets\\Test-Dataset\\DARK FACE\\edge\\10.png',
        'H:\\datasets\\Test-Dataset\\DARK FACE\\edge\\15.png'
    ]

    gt_paths = [
        'H:\\datasets\\Test-Dataset\\DARK FACE\\Normal\\0.png',
        'H:\\datasets\\Test-Dataset\\DARK FACE\\Normal\\5.png',
        'H:\\datasets\\Test-Dataset\\DARK FACE\\Normal\\10.png',
        'H:\\datasets\\Test-Dataset\\DARK FACE\\Normal\\15.png'
    ]

    # 创建子图
    fig, axs = plt.subplots(4, 3, figsize=(12, 16))

    for i in range(4):
        img_path = image_paths[i]
        edge_path = edge_paths[i]
        gt_path = gt_paths[i]

        # 获取预测结果
        predicted_image = predict(model, img_path, edge_path, transform)

        # 原图
        img = Image.open(img_path)
        axs[i, 0].imshow(img)
        axs[i, 0].set_title(f"Low-Light Image {i + 1}")
        axs[i, 0].axis('off')

        # 预测图
        axs[i, 1].imshow(predicted_image.permute(1, 2, 0))  # 转换为 HWC 格式以适配imshow
        axs[i, 1].set_title(f"Predicted Image {i + 1}")
        axs[i, 1].axis('off')

        # 边缘图
        edge = Image.open(gt_path)
        axs[i, 2].imshow(edge)
        axs[i, 2].set_title(f"GT Image {i + 1}")
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

    print("Evaluation Metrics:")
    test_loader = load_dataset(dataset_paths, transform)
    # ssim, psnr, lpips, nima, loe = evaluate_model(model, test_loader, args)
    # print(f"SSIM: {ssim:.4f}", f"PSNR: {psnr:.4f}", f"LPIPS: {lpips:.4f}", f"NIMA: {nima:.4f}", f"LOE: {loe:.4f}")
    evaluate_model(model, test_loader, args)

if __name__ == '__main__':
    main()
