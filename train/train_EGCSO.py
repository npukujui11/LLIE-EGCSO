# Copyright(c) 2024
# Licensed under the MIT License
# by @npukujui11
# 用于训练EGCSO模型

import os
import sys

# 添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from models.EGCSO import EGCSO
from torchvision.transforms import transforms, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from tqdm import tqdm
from models.loss.lossfunction import *


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据预处理
transform = transforms.Compose([
    RandomCrop(256),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    ])

class LowLightDataset(Dataset):
    def __init__(self, img_dir, normal_dir, edge_dir, transform=None):
        self.img_dir = img_dir
        self.normal_dir = normal_dir
        self.edge_dir = edge_dir

        self.img_files = os.listdir(img_dir)
        self.normal_files = os.listdir(normal_dir)
        self.edge_files = os.listdir(edge_dir)

        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        normal_path = os.path.join(self.normal_dir, self.normal_files[idx])
        edge_path = os.path.join(self.edge_dir, self.edge_files[idx])

        # 用PIL.Image打开图像并转换成Tensor
        img = Image.open(img_path).convert('RGB')
        normal = Image.open(normal_path).convert('RGB')
        edge = Image.open(edge_path).convert('L')

        if self.transform:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            img = self.transform(img)
            torch.manual_seed(seed)
            normal = self.transform(normal)
            torch.manual_seed(seed)
            edge = self.transform(edge)

        return img, normal, edge


def train(model, train_loader, criterion, optimizer, scheduler, checkpoint_path, start_epochs=0, num_epochs=100):
    model.train()
    train_losses = []

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    loss_log_path = os.path.join(checkpoint_dir, "EGCSO.txt")
    if not os.path.exists(loss_log_path):
        with open(loss_log_path, mode='w') as loss_file:
            loss_file.write("Epoch\tLoss\tEpoch Avg Loss\n")  # 写入表头

    for epoch in range(start_epochs, num_epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader, leave=True, desc=f"Epoch {epoch + 1}/{num_epochs}", ascii=True, ncols=100)

        for i, (img, normal, edge) in enumerate(pbar):
            img, normal, edge = img.to(device), normal.to(device), edge.to(device)

            # 前向传播
            optimizer.zero_grad()
            output = model(img, edge)
            output_hvi = model.HVIT(output, edge)
            normal_hvi = model.HVIT(normal, edge)

            ######################################
            # 1.
            # If you do not want to use PERCEPTUAL LOSS (P_loss)
            # comment the following code
            ######################################
            L1_loss = L1Loss(loss_weight=0, reduction='mean').to(device)
            D_loss = SSIM(weight=0).to(device)
            E_loss = EdgeLoss(loss_weight=0).to(device)
            P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1, 'conv3_4': 1, 'conv4_4': 1},
                                    perceptual_weight=1.0, criterion='mse').to(device)

            loss_hvi = (L1_loss(output_hvi, normal_hvi) + D_loss(output_hvi, normal_hvi) +
                            E_loss(output_hvi, normal_hvi) + 1.0 * P_loss(output_hvi, normal_hvi)[0])
            loss_rgb = (L1_loss(output, normal) + D_loss(output, normal) +
                            E_loss(output, normal) + 1.0 * P_loss(output, normal)[0])
            loss = loss_rgb + 1.0 * loss_hvi
            ######################################
            # 2.
            # Uncomment the following line
            ######################################
            # loss = criterion(output, normal) + 1.0 * criterion(output_hvi, normal_hvi)
            epoch_loss += loss.item()

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

        # Every epoch, save the model weights
        torch.save(model.state_dict(), checkpoint_path)

        scheduler.step()
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {epoch_loss:.4f}")
        # 保存损失到txt文件
        with open(loss_log_path, mode='a') as loss_file:
            loss_file.write(f"{epoch + 1}\t{epoch_loss:.4f}\n")

    return train_losses


dataset_paths = {
    'train_image': '../datasets/LOLv2/Synthetic/Train/Low',
    'train_normal': '../datasets/LOLv2/Synthetic/Train/Normal',
    'train_edge': '../datasets/LOLv2/Synthetic/Train/Normal_edge',

    'test_image': '../datasets/LOLv2/Synthetic/Test/Low',
    'test_normal': '../datasets/LOLv2/Synthetic/Test/Normal',
    'test_edge': '../datasets/LOLv2/Synthetic/Test/Normal_edge'
}


def main():
    # 设置路径
    img_dir, normal_dir, edge_dir = dataset_paths['train_image'], dataset_paths['train_normal'], dataset_paths['train_edge']

    # 超参数
    batch_size = 8

    # 加载数据集
    train_dataset = LowLightDataset(img_dir, normal_dir, edge_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 初始化模型
    model = EGCSO().to(device)

    # 定义损失函数和优化器
    criterion = lambda output, normal: 0.6 * nn.MSELoss()(output, normal) + 0.4 * SSIM()(output, normal)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)


    # 训练模型
    checkpoint_path = '../checkpoints/EGCSO/EGCSO_ep2000.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()

    train_losses = train(model, train_loader, criterion, optimizer, scheduler,
                         checkpoint_path, start_epochs=1500, num_epochs=2000)


if __name__ == '__main__':
    main()
