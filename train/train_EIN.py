# Copyright (c) 2024
# Licensed under the MIT License
# by @npukujui11
# 用于训练 EIN 模型，EIN 模型用于从弱光中直接捕获边缘结构信息
# 训练策略：使用BDCN的loss函数，训练2000个epoch，batch_size=32
# 优化器：Adam，学习率1e-4，权重衰减1e-8

import os
import sys

# 获取当前脚本的目录（train 文件夹）
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上一级目录（D:\program\LLIE\DEN）
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# 添加到 sys.path
sys.path.append(parent_dir)

import json
import time

import argparse
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import *
from models.EIN import *
from models.loss.lossfunction import *


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据预处理
transform = transforms.Compose([
    RandomCrop(256),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),

    # RandomResizedCrop(256, scale=(0.8, 1.0)), # 随机缩放
    # RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # 随机仿射变换
    # ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # 色彩增强
    # RandomGrayscale(p=0.2),  # 随机灰度
    # RandomRotation(degrees=(-10, 10)),  # 小角度旋转
    # GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),  # 模糊增强

    transforms.ToTensor(),
    ])


def parse_args():
    parser = argparse.ArgumentParser(description="Train EIN model")
    parser.add_argument('--config', type=str, default='../train/config.json', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/EdgeNet/EIN_wde_wBDCN_op_ep150.pth',
                        help='Path to save checkpoint')
    return parser.parse_args()


class EdgeDataset(Dataset):
    def __init__(self, imgs_dir, edge_maps_dir, transform=None, img_suffix=".jpg", edge_suffix=".png"):
        self.imgs_dir = imgs_dir
        self.edge_maps_dir = edge_maps_dir
        self.transform = transform
        self.img_suffix = img_suffix
        self.edge_suffix = edge_suffix
        self.filenames = os.listdir(imgs_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        imgs_filename = self.filenames[idx]
        edge_maps_filename = imgs_filename.replace(self.img_suffix, self.edge_suffix)

        imgs_path = os.path.join(self.imgs_dir, imgs_filename)
        edge_maps_path = os.path.join(self.edge_maps_dir, edge_maps_filename)

        imgs_image = Image.open(imgs_path).convert('RGB')
        edge_maps_image = Image.open(edge_maps_path).convert('L')

        if self.transform:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            imgs_image = self.transform(imgs_image)
            torch.manual_seed(seed)
            edge_maps_image = self.transform(edge_maps_image)

        return imgs_image, edge_maps_image, imgs_filename


'''
dataset_paths = {
    'train_image': '../datasets/BIPEDv2/Train/imgs_aug',
    'train_edge': '../datasets/BIPEDv2/Train/edges_aug',
    'test_image': '../datasets/BIPEDv2/Test/imgs_aug',
    'test_edge': '../datasets/BIPEDv2/Test/edges_aug'
}
'''

"""
dataset_paths = {
    'train_image': '../datasets/BIPEDv2/Train/imgs_degraded',
    'train_edge': '../datasets/BIPEDv2/Train/edges',
    'test_image': '../datasets/BIPEDv2/Test/imgs_degraded',
    'test_edge': '../datasets/BIPEDv2/Test/edges'
}
"""

"""
dataset_paths = {
    'train_image': '../datasets/LOL-v2/Real_captured/Train/Low',
    'train_edge': '../datasets/LOL-v2/Real_captured/Train/Normal_edge',
    'test_image': '../datasets/LOL-v2/Real_captured/Test/Low',
    'test_edge': '../datasets/LOL-v2/Real_captured/Test/Normal_edge'
}
"""


dataset_paths = {
    'train_image': '../datasets/LOL-v2/Real_captured/Train/Low_aug',
    'train_edge': '../datasets/LOL-v2/Real_captured/Train/Normal_edge_aug',
    'test_image': '../datasets/LOL-v2/Real_captured/Test/Low_aug',
    'test_edge': '../datasets/LOL-v2/Real_captured/Test/Normal_edge_aug'
}


def load_dataset(paths, transform, batch_size=32, num_workers=1):
    train_dataset = EdgeDataset(paths['train_image'], paths['train_edge'], transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = EdgeDataset(paths['test_image'], paths['test_edge'], transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader



def train_model(model, criterion, optimizer, scheduler, train_loader, checkpoint_path, start_epochs=0, num_epochs=20):
    model.train()
    train_losses = []

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    loss_log_path = os.path.join(checkpoint_dir, 'EIN_wde_wBDCN_op.txt')
    if not os.path.exists(loss_log_path):
        with open(loss_log_path, mode='w') as loss_file:
            loss_file.write("Epoch\tloss\tEpoch Avg Loss\n") # 写入表头

    start_time = time.time()  # Record start time

    for epoch in range(start_epochs, num_epochs):
        model.train()
        epoch_loss = 0
        l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 1.3] # new BDC loss
        # l_weight = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.1]  # for bdcn ori loss
        # l_weight = [[0.05, 2.], [0.05, 2.], [0.05, 2.],
        #             [0.1, 1.], [0.1, 1.], [0.1, 1.],
        #             [0.01, 4.]]  # for cats loss
        pbar = tqdm(train_loader, leave=True, desc=f"Epoch {epoch + 1}/{num_epochs}", ascii=True, ncols=100)

        for i, (image, edge, filename) in enumerate(pbar):
            image, edge = image.to(device), edge.to(device)
            optimizer.zero_grad()

            preds_list = model(image)
            # loss = sum([criterion(preds, edge,l_w) for preds, l_w in zip(preds_list,l_weight)]) # bdcn_loss
            fidelity_loss = sum([criterion(preds_list[-1], edge, l_weight[-1])])  # bdcn_loss
            consistency_loss = criterion(preds_list[-1], edge)
            loss = fidelity_loss + consistency_loss
            # loss = sum([criterion(preds, edge, l_w, device) for preds, l_w in zip(preds_list, l_weight)])  # cats_loss
            # loss = sum([criterion(preds_list[-1], edge, l_weight[-1], device)])  # cats_loss
            # loss = sum([criterion(preds, labels) for preds in preds_list])  #HED loss, rcf_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})
            epoch_loss += loss.item()

        try:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"模型成功保存到 {checkpoint_path}")
        except Exception as e:
            print(f"模型保存失败: {e}")

        scheduler.step()
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")

        with open(loss_log_path, mode='a') as loss_file:
            loss_file.write(f"{epoch + 1}\t{epoch_loss:.4f}\n")

    # Record end time
    end_time = time.time()
    train_time = end_time - start_time

    return train_losses, train_time


def plot_loss_curve(train_losses, num_epochs):
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()


def main():
    args = parse_args()

    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    start_epochs = config['start_epochs']

    train_loader, test_loader = load_dataset(dataset_paths, transform, batch_size, num_workers)

    model = EIN().to(device)

    # Loss function and optimizer
    criterion = bdcn_loss2
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    # Train the model
    checkpoint_path = args.checkpoint
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()

    train_losses, train_time = train_model(model, criterion, optimizer, scheduler, train_loader,
                                           checkpoint_path, start_epochs, num_epochs)

    print("Finished Training")

    # Save the updated config to config.json
    with open('config.json', 'w') as config_file:
        json.dump(config, config_file)

    # Plot the loss curve
    plot_loss_curve(train_losses, num_epochs)

    print("Training Time: ", train_time, " seconds")

if __name__ == '__main__':
    main()
