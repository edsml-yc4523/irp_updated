"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from albumentations import Compose, HorizontalFlip, VerticalFlip, Rotate


def get_aug():
    """
    Returns a composition of data augmentation transformations.

    # The augmentations include horizontal and vertical flips,
    # as well as rotations.

    Returns:
        Compose: An albumentations Compose object containing the augmentations.
    """
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Rotate(limit=15, p=0.5)
    ], is_check_shapes=False)


def compute_mean_std(directory):
    # Compute dataset mean and std
    pixel_num = 0
    channel_sum = None
    channel_sum_squared = None

    def process_image(img_path):
        nonlocal pixel_num, channel_sum, channel_sum_squared
        img = np.load(img_path)
        if channel_sum is None:
            channel_sum = np.zeros(img.shape[2])
            channel_sum_squared = np.zeros(img.shape[2])
        pixel_num += (img.shape[0] * img.shape[1])
        channel_sum += np.sum(img, axis=(0, 1))
        channel_sum_squared += np.sum(
            np.square(img, dtype=np.float64), axis=(0, 1)
            )

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                process_image(os.path.join(root, file))

    mean = channel_sum / pixel_num
    variance = channel_sum_squared / pixel_num - np.square(mean)
    std = np.sqrt(variance.clip(min=0))
    return mean, std


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size, color_to_label, 
                 mean, std, augmentations=None):
        """
        Initialize the segmentation dataset.

        Args:
            images_dir (str): Path to the directory containing images.
            masks_dir (str): Path to the directory containing masks.
            img_size (tuple): Target image size (width, height).
            color_to_label (dict): Dictionary mapping colors to labels.
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.
            augmentations (albumentations.Compose, optional): Data augmentation
            pipeline. Defaults to None.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.color_to_label = color_to_label
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))
        self.augmentations = (augmentations if augmentations is not None
                              else get_aug())
        self.mean = torch.tensor(mean).view(6, 1, 1)
        self.std = torch.tensor(std).view(6, 1, 1)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Get a data sample.

        Args:
            idx (int): Index of the data sample.

        Returns:
            tuple: A tuple containing the image and corresponding label.
        """
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, 
                                 self.mask_files[idx].replace('.npy', '.png'))

        # Load the image and mask
        image = np.load(img_path)
        mask = np.array(Image.open(mask_path))

        # Apply augmentations if any
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert to Tensor
        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        # Center crop and resize
        center_crop = transforms.CenterCrop((116, 116))
        image = center_crop(image)
        mask = center_crop(mask)

        resize = transforms.Resize(self.img_size)
        image = resize(image)
        mask = resize(mask)

        # Normalize the image
        image = (image - self.mean) / self.std

        # Convert mask to labels
        mask = self.mask_to_label(mask.numpy())

        return image, mask

    def mask_to_label(self, mask):
        """
        Convert color mask to label mask.

        Args:
            mask (numpy.ndarray): Color mask with shape (H, W, C).

        Returns:
            torch.Tensor: Label mask with shape (H, W).
        """
        mask = np.transpose(mask, (1, 2, 0))
        label_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        for color, label in self.color_to_label.items():
            matches = np.all(mask == color, axis=-1)
            label_mask[matches] = label
        return torch.from_numpy(label_mask).long()
