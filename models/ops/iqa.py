# Copyright (c) 2024
# Licensed under the MIT License
# by @npukujui11
# Image Quality Assessment (IQA) Metrics

import cv2
import os
from PIL import Image
import pyiqa
import lpips
import torch
import numpy as np
from torchvision.transforms import ToTensor
from piq import brisque
import torch.nn.functional as F
from tqdm import tqdm

from skimage.transform import resize
from pytorch_msssim import ssim as calculate_ssim
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import binarize
from basicsr.metrics.niqe import calculate_niqe
from brisque import BRISQUE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

loss_fn_alex = lpips.LPIPS(net='alex').to(device)

nima_model = pyiqa.create_metric('nima', device=device)

def calculate_brisque(img):
    ndarray = np.asarray(img)
    obj = BRISQUE(url=False)
    brisque_score = obj.score(img=ndarray)
    return brisque_score

# Calculate LPIPS
def calculate_lpips(img1, img2):

    img1_tensor = (img1.float() - img1.min())/ (img1.max()-img1.min())
    img2_tensor = (img2.float() - img1.min())/ (img1.max()-img1.min())

    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)

    lpips_score = loss_fn_alex.forward(img1_tensor, img2_tensor).mean()

    return lpips_score


# Calculate NIMA
def calculate_nima(img):

    img_tensor = (img - img.min()) / (img.max() - img.min())
    img_tensor = img_tensor.to(device)

    nima_score = nima_model(img_tensor).mean()

    return nima_score


# Calculate LOE
def calculate_loe(img1, img2):
    # Get the dimensions of the images
    B, C, H, W = img1.shape

    # Convert to numpy arrays and move channels to last dimension
    img1 = img1.permute(0, 2, 3, 1).cpu().numpy()
    img2 = img2.permute(0, 2, 3, 1).cpu().numpy()

    loe_score = []

    for i in range(B):
        # Process each image pair in the batch
        L = np.max(img1[i], axis=0)
        Le = np.max(img2[i], axis=0)

        # Resize the images
        r = 50 / min(W, H)
        Md = round(W * r)
        Nd = round(H * r)
        Ld = resize(L, (Nd, Md), anti_aliasing=True)
        Led = resize(Le, (Nd, Md), anti_aliasing=True)

        RD = np.zeros((Nd, Md))

        for y in range(Md):
            for x in range(Nd):
                E = np.logical_xor((Ld[x, y] >= Ld), (Led[x, y] >= Led))
                RD[x, y] = np.sum(E)

        loe_value = np.mean(RD) / (Md * Nd)
        loe_score.append(loe_value)

    # Calculate the average LOE score over the batch
    avg_loe_score = np.mean(loe_score)
    return avg_loe_score

def calculate_NIQE(img):
    img = np.array(img)
    img_np = img.astype(np.float32)

    niqe_score = calculate_niqe(img_np, 4, 'HWC', 'y')
    return niqe_score

# 计算PSNR
def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    psnr = 10 * torch.log10(1 / mse)
    return psnr


# 加载图像并转换为 Tensor
def load_image_as_tensor(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = ToTensor()(img).unsqueeze(0)  # 添加 batch 维度
    return img_tensor

# 主函数：计算文件夹中所有图像的指标
def evaluate_images(test_folder, ref_folder):
    test_images = sorted(os.listdir(test_folder))
    ref_images = sorted(os.listdir(ref_folder))

    # 初始化累加器
    total_brisque = 0.0
    total_lpips = 0.0
    total_nima = 0.0
    total_loe = 0.0
    total_niqe = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_images = len(test_images)
    failed_images = []

    for test_img_name, ref_img_name in tqdm(zip(test_images, ref_images), total=num_images, desc="Processing images"):
        test_img_path = os.path.join(test_folder, test_img_name)
        ref_img_path = os.path.join(ref_folder, ref_img_name)

        try:
            # 加载图像
            test_img = Image.open(test_img_path).convert('RGB')
            ref_img = Image.open(ref_img_path).convert('RGB')

            if test_img.size != ref_img.size:
                print(f"Resizing test image: {test_img_name} to match reference image: {ref_img_name}")
                test_img = test_img.resize(ref_img.size, Image.ANTIALIAS)

            # 转换为 Tensor
            test_img_tensor = ToTensor()(test_img).unsqueeze(0).to(device)
            ref_img_tensor = ToTensor()(ref_img).unsqueeze(0).to(device)

            # 计算指标
            brisque_score = calculate_brisque(test_img)
            lpips_score = calculate_lpips(test_img_tensor, ref_img_tensor)
            nima_score = calculate_nima(test_img_tensor)
            loe_score = calculate_loe(test_img_tensor, ref_img_tensor)
            niqe_score = calculate_NIQE(test_img)
            psnr_value = calculate_psnr(test_img_tensor, ref_img_tensor)
            ssim_value = calculate_ssim(test_img_tensor, ref_img_tensor, data_range=1.0, size_average=True)

            # 累加指标
            total_brisque += brisque_score
            total_lpips += lpips_score
            total_nima += nima_score
            total_loe += loe_score
            total_niqe += niqe_score
            total_psnr += psnr_value
            total_ssim += ssim_value

            # 打印当前图像的指标
            print(f"Image: {test_img_name}")
            print(f"BRISQUE: {brisque_score:.4f}")
            print(f"LPIPS: {lpips_score:.4f}")
            print(f"NIMA: {nima_score:.4f}")
            print(f"LOE: {loe_score:.4f}")
            print(f"NIQE: {niqe_score:.4f}")
            print(f"PSNR: {psnr_value:.4f}")
            print(f"SSIM: {ssim_value:.4f}")
            print("-" * 40)
        except Exception as e:
            print(f"Error processing image: {test_img_name}: {e}")
            failed_images.append(test_img_name)
            continue

    # 计算平均指标
    avg_brisque = total_brisque / num_images
    avg_lpips = total_lpips / num_images
    avg_nima = total_nima / num_images
    avg_loe = total_loe / num_images
    avg_niqe = total_niqe / num_images
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images

    # 打印平均指标
    print("Average Metrics:")
    print(f"Average BRISQUE: {avg_brisque:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    print(f"Average NIMA: {avg_nima:.4f}")
    print(f"Average LOE: {avg_loe:.4f}")
    print(f"Average NIQE: {avg_niqe:.4f}")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")

    # 输出无法处理的图片文件名
    if failed_images:
        print("\nFailed to process the following images:")
        for img_name in failed_images:
            print(img_name)

    return {
        'avg_brisque': avg_brisque,
        'avg_lpips': avg_lpips,
        'avg_nima': avg_nima,
        'avg_loe': avg_loe,
        'avg_niqe': avg_niqe,
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'failed_images': failed_images
    }

# 示例用法
if __name__ == "__main__":
    test_folder = ("G:\\模型复现结果\\√（2022）Bread\√SICE")  # 替换为待评估图像文件夹路径
    ref_folder = ("H:\\datasets\\SICE\\Normal")  # 替换为参考图像文件夹路径

    results = evaluate_images(test_folder, ref_folder)



# Calculate ODS score
# ODS 是在整个数据集上计算的最佳阈值时的F1得分。
def calculate_ods(predictions, ground_truths, threshold_steps=100):

    thresholds = np.linspace(0, 1, threshold_steps)
    max_f1_score = 0

    for threshold in thresholds:
        binary_preds = (predictions >= threshold).astype(np.float32)
        tp = np.sum((binary_preds == 1) & (ground_truths == 1))
        fp = np.sum((binary_preds == 1) & (ground_truths == 0))
        fn = np.sum((binary_preds == 0) & (ground_truths == 1))

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (fn + tp + 1e-10)

        f1_score = 2 * precision * recall / (precision + recall + 1e-10)
        if f1_score > max_f1_score:
            max_f1_score = f1_score

    return max_f1_score


# Calculate OIS score
# OIS 是在每个图像上计算最佳阈值时的F1得分，然后在整个数据集上取平均值。
def calculate_ois(predictions, ground_truths, threshold_steps=100):
    thresholds = np.linspace(0, 1, threshold_steps)
    f1_scores = []

    for i in range(predictions.shape[0]):
        max_f1_score = 0
        for threshold in thresholds:
            binary_pred = (predictions[i] >= threshold).astype(np.float32)
            tp = np.sum((binary_pred == 1) & (ground_truths[i] == 1))
            fp = np.sum((binary_pred == 1) & (ground_truths[i] == 0))
            fn = np.sum((binary_pred == 0) & (ground_truths[i] == 1))

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (fn + tp + 1e-10)

            f1_score = 2 * precision * recall / (precision + recall + 1e-10)
            if f1_score > max_f1_score:
                max_f1_score = f1_score

        f1_scores.append(max_f1_score)

    return np.mean(f1_scores)


# Calculate AP score
# AP 是精确率-召回率曲线下的面积。
def calculate_ap(predictions, ground_truths):
    precisions, recalls = [], []
    thresholds = np.linspace(0, 1, 100)

    for threshold in thresholds:
        binary_preds = (predictions >= threshold).astype(np.float32)
        tp = np.sum((binary_preds == 1) & (ground_truths == 1))
        fp = np.sum((binary_preds == 1) & (ground_truths == 0))
        fn = np.sum((binary_preds == 0) & (ground_truths == 1))

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (fn + tp + 1e-10)

        precisions.append(precision)
        recalls.append(recall)

    precisions = np.array(precisions)
    recalls = np.array(recalls)

    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]

    ap = 0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]

    return ap
