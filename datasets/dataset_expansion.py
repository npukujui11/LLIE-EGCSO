# Copyright (c) 2024
# Licensed under the MIT License
# by @npkujui11
# 用于拓展边缘数据集的大小，包括分割图像，旋转图像，翻转图像等

import cv2
import os
import numpy as np
import random
from glob import glob


def split_image(image):
    height, width = image.shape[:2]
    return image[:, :width // 2], image[:, width // 2:]


def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (width, height))
    return rotated


def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def augment_image(image, label):
    augmented_images = []
    augmented_labels = []

    # Split image
    img1, img2 = split_image(image)
    lbl1, lbl2 = split_image(label)

    # Rotate and crop
    for img, lbl in zip([img1, img2], [lbl1, lbl2]):
        for angle in np.linspace(0, 360, 15, endpoint=False):
            rotated_img = rotate_image(img, angle)
            rotated_lbl = rotate_image(lbl, angle)
            cropped_img = rotated_img[::]
            cropped_lbl = rotated_lbl[::]
            augmented_images.append(cropped_img)
            augmented_labels.append(cropped_lbl)

    # Horizontal flip
    flipped_images = [cv2.flip(img, 1) for img in augmented_images]
    flipped_labels = [cv2.flip(lbl, 1) for lbl in augmented_labels]
    augmented_images.extend(flipped_images)
    augmented_labels.extend(flipped_labels)

    # Gamma correction
    gamma_corrected_images = []
    gamma_corrected_labels = []

    for img, lbl in zip(augmented_images, augmented_labels):
        gamma_corrected_images.append(gamma_correction(img, 0.3030))
        gamma_corrected_images.append(gamma_correction(img, 0.6060))
        gamma_corrected_labels.append(lbl)
        gamma_corrected_labels.append(lbl)

    augmented_images.extend(gamma_corrected_images)
    augmented_labels.extend(gamma_corrected_labels)

    return augmented_images, augmented_labels


def process_dataset(input_dir, label_dir, output_dir, output_label_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # 获取所有图像文件
    image_paths = glob(os.path.join(input_dir, '*.png'))  # 根据需要调整图像格式
    label_paths = glob(os.path.join(label_dir, '*.png'))  # 根据需要调整图像格式

    image_paths.sort()
    label_paths.sort()

    for img_path, lbl_path in zip(image_paths, label_paths):
        img = cv2.imread(img_path)
        lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        augmented_images, augmented_labels = augment_image(img, lbl)

        base_name = os.path.basename(img_path).split('.')[0]
        for i, (aug_img, aug_lbl) in enumerate(zip(augmented_images, augmented_labels)):
            aug_img_path = os.path.join(output_dir, f"{base_name}_aug_{i}.png")
            aug_lbl_path = os.path.join(output_label_dir, f"{base_name}_aug_{i}.png")
            cv2.imwrite(aug_img_path, aug_img)
            cv2.imwrite(aug_lbl_path, aug_lbl)

        print(f"Processed {base_name}: {len(augmented_images)} augmented images generated.")


train_input_image_dir = '../datasets/LOL-v2/Real_captured/Train/Low'
train_input_label_dir = '../datasets/LOL-v2/Real_captured/Train/Normal_edge'
train_output_image_dir = '../datasets/LOL-v2/Real_captured/Train/Low_aug'
train_output_label_dir = '../datasets/LOL-v2/Real_captured/Train/Normal_edge_aug'

test_input_image_dir = '../datasets/LOL-v2/Real_captured/Test/Low'
test_input_label_dir = '../datasets/LOL-v2/Real_captured/Test/Normal_edge'
test_output_image_dir = '../datasets/LOL-v2/Real_captured/Test/Low_aug'
test_output_label_dir = '../datasets/LOL-v2/Real_captured/Test/Normal_edge_aug'

# if folder does not exist, create it
if not os.path.exists(train_output_image_dir):
    os.makedirs(train_output_image_dir)
if not os.path.exists(train_output_label_dir):
    os.makedirs(train_output_label_dir)
if not os.path.exists(test_output_image_dir):
    os.makedirs(test_output_image_dir)
if not os.path.exists(test_output_label_dir):
    os.makedirs(test_output_label_dir)

# Process training dataset
print("Processing training dataset...")
process_dataset(train_input_image_dir, train_input_label_dir, train_output_image_dir, train_output_label_dir)

# Process test dataset
print("Processing test dataset...")
process_dataset(test_input_image_dir, test_input_label_dir, test_output_image_dir, test_output_label_dir)

print("Dataset augmentation completed.")
