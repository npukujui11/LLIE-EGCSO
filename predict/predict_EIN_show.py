# Copyright (c) 2024
# Licensed under the MIT License
# by @npukujui11
# 用于预测 EIN 模型，EIN 模型用于从弱光中直接捕获边缘结构信息。因此预测的结果是边缘结构图像
# 但是基于深度学习的方法，预测的结果是一个概率图，也就是边缘相对比较宽。因此需要将概率图转换为二值图
# 需要用到 NMS（非极大值抑制）方法

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
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
from models.EIN import DexiNed
from datasets.dataloader import *
from torch.utils.data import DataLoader
from models.ops.iqa import calculate_ods, calculate_ois, calculate_ap
from torchvision.transforms import transforms
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_paths = {
    'train_image': '../datasets/LOL-v2/Real_captured/Train/Low',
    'train_edge': '../datasets/LOL-v2/Real_captured/Train/Normal_edge',
    'test_image': '../datasets/LOL-v2/Real_captured/Test/Low',
    'test_edge': '../datasets/LOL-v2/Real_captured/Test/Normal_edge'
}

"""
dataset_paths = {
    'train_image': '../datasets/BIPEDv2/Train/imgs_degraded',
    'train_edge': '../datasets/BIPEDv2/Train/edges',
    'test_image': '../datasets/BIPEDv2/Test/imgs_degraded',
    'test_edge': '../datasets/BIPEDv2/Test/edges'
}
"""

def load_dataset(paths, transform, batch_size=32, num_workers=1):
    test_dataset = EdgeDataset(paths['test_image'], paths['test_edge'], transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return test_loader


# Image Preprocessing
transform = transforms.Compose([transforms.ToTensor()])


def plot_predictions(images, predictions, targets, save_path=None):
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))
    for i in range(4):
        axes[i, 0].imshow(images[i])  # Input image
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(predictions[i], cmap='gray')  # Predicted edge map
        axes[i, 1].set_title('Predicted Edges')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(targets[i], cmap='gray')  # Ground truth edge map
        axes[i, 2].set_title('Ground Truth Edges')
        axes[i, 2].axis('off')

    plt.suptitle('Visualization Results', fontsize=25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def evaluate_model(model, test_loader, save_dir):
    # Save directory
    os.makedirs(save_dir, exist_ok=True)

    # Evaluation mode
    model.eval()
    total_duration = []
    pbar = tqdm(test_loader, desc="Evaluating", ascii=True, ncols=100)

    with torch.no_grad():
        for i, (images, edge, file_names) in enumerate(pbar):
            images, edge = images.to(device), edge.to(device)

            # Timing the inference
            start_time = time.perf_counter()
            outputs = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            duration = time.perf_counter() - start_time
            total_duration.append(duration)

            # Save the last prediction as PNG
            prediction = torch.sigmoid(outputs[-1]).cpu().numpy()

            # Convert to 8-bit image
            prediction = (prediction * 255).astype(np.uint8)

            # Ensure the prediction is in the right shape
            if prediction.ndim == 3 and prediction.shape[0] == 1:
                prediction = prediction[0]  # Remove the channel dimension if it is 1
            elif prediction.ndim == 4 and prediction.shape[1] == 1:
                prediction = prediction[:, 0]  # Remove the channel dimension if it is 1

            # Save each prediction image
            for j in range(prediction.shape[0]):
                prediction_image = Image.fromarray(prediction[j])  # Assuming single channel

                # Get the original file name without extension
                original_name = os.path.splitext(file_names[j])[0]
                # Create the prediction file name
                save_path = os.path.join(save_dir, f"{original_name}.png")
                prediction_image.save(save_path)

            # Save the predictions and ground truth
            predictions = [torch.sigmoid(output).cpu().numpy() for output in outputs]
            gt = edge.cpu().numpy()

    ods_score = calculate_ods(np.array(predictions[-1]), np.array(gt))
    ois_score = calculate_ois(np.array(predictions[-1]), np.array(gt))
    ap_score = calculate_ap(np.array(predictions[-1]), np.array(gt))

    return total_duration, ods_score, ois_score, ap_score


def main():
    # Data transformation
    transform = transforms.Compose([transforms.ToTensor()])

    # Data Loader
    test_loader = load_dataset(dataset_paths, transform, batch_size=16, num_workers=1)

    # Load Model
    model = DexiNed().to(device)

    checkpoint_path = '../checkpoints/EdgeNet/EIN_wde_wBDCN_op.pth'
    if os.path.exists(checkpoint_path):
        print(f"Restoring weights from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    total_duration, ods_score, ois_score, ap_score = evaluate_model(model, test_loader,
                                                                    '../datasets/LOL-v2/Real_captured/Test/predictions')

    # Print evaluation metrics
    print(f"Test ODS: {ods_score:.4f}, Test OIS: {ois_score:.4f}, Test AP: {ap_score:.4f}, "
          f"Inference Time: {np.mean(total_duration):.4f} sec")
    # print(f"Inference Time: {np.mean(total_duration):.4f} sec")

    # Image and edge paths
    image_paths = [
        '../datasets/LOL-v2/Real_captured/Test/Low/low00690.png',
        '../datasets/LOL-v2/Real_captured/Test/Low/low00692.png',
        '../datasets/LOL-v2/Real_captured/Test/Low/low00694.png',
        '../datasets/LOL-v2/Real_captured/Test/Low/low00696.png'
        ''
    ]
    edge_paths = [
        '../datasets/LOL-v2/Real_captured/Test/Normal_edge/low00690.png',
        '../datasets/LOL-v2/Real_captured/Test/Normal_edge/low00692.png',
        '../datasets/LOL-v2/Real_captured/Test/Normal_edge/low00694.png',
        '../datasets/LOL-v2/Real_captured/Test/Normal_edge/low00696.png'
    ]

    """
    image_paths = [
        '../datasets/BIPEDv2/Test/imgs_degraded/RGB_008.jpg',
        '../datasets/BIPEDv2/Test/imgs_degraded/RGB_010.jpg',
        '../datasets/BIPEDv2/Test/imgs_degraded/RGB_017.jpg',
        '../datasets/BIPEDv2/Test/imgs_degraded/RGB_017.jpg'
        ''
    ]
    edge_paths = [
        '../datasets/BIPEDv2/Test/edges/RGB_008.png',
        '../datasets/BIPEDv2/Test/edges/RGB_010.png',
        '../datasets/BIPEDv2/Test/edges/RGB_ 017.png',
        '../datasets/BIPEDv2/Test/edges/RGB_025.png'
    ]
    """

    images, edges, predictions = [], [], []

    with torch.no_grad():
        for img_path, edge_path in zip(image_paths, edge_paths):
            image = Image.open(img_path)
            edge = Image.open(edge_path)

            images.append(image)
            edges.append(edge)

            img_tensor = transform(image).unsqueeze(0).to(device)
            pred_tensor = model(img_tensor)
            prediction = torch.sigmoid(pred_tensor[-1]).squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)

            predictions.append(prediction)

    plot_predictions(images, predictions, edges)


if __name__ == '__main__':
    main()
