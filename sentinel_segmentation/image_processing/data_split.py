"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import os
import shutil
from sklearn.model_selection import train_test_split


def split_dataset(data_dir, output_dir, test_size=0.3, val_size=1/3):
    """
    Split dataset into training, validation, and testing sets.

    Args:
        data_dir (str): Path to the dataset directory.
        output_dir (str): Path to save the split datasets.
        test_size (float): Proportion of the dataset
                            to include in the test split.
        val_size (float): Proportion of the test dataset
                            to include in the validation split.
    """
    images_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'masks')

    train_images_dir = os.path.join(output_dir, 'train/images')
    train_masks_dir = os.path.join(output_dir, 'train/masks')
    val_images_dir = os.path.join(output_dir, 'val/images')
    val_masks_dir = os.path.join(output_dir, 'val/masks')
    test_images_dir = os.path.join(output_dir, 'test/images')
    test_masks_dir = os.path.join(output_dir, 'test/masks')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_masks_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_masks_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_masks_dir, exist_ok=True)

    image_files = sorted(os.listdir(images_dir))
    mask_files = sorted(os.listdir(masks_dir))

    train_images, test_images, train_masks, test_masks = train_test_split(
        image_files, mask_files, test_size=test_size, random_state=42
    )
    val_images, test_images, val_masks, test_masks = train_test_split(
        test_images, test_masks, test_size=val_size, random_state=42
    )

    for img_file, mask_file in zip(train_images, train_masks):
        shutil.move(os.path.join(images_dir, img_file),
                    os.path.join(train_images_dir, img_file))
        shutil.move(os.path.join(masks_dir, mask_file),
                    os.path.join(train_masks_dir, mask_file))

    for img_file, mask_file in zip(val_images, val_masks):
        shutil.move(os.path.join(images_dir, img_file),
                    os.path.join(val_images_dir, img_file))
        shutil.move(os.path.join(masks_dir, mask_file),
                    os.path.join(val_masks_dir, mask_file))

    for img_file, mask_file in zip(test_images, test_masks):
        shutil.move(os.path.join(images_dir, img_file),
                    os.path.join(test_images_dir, img_file))
        shutil.move(os.path.join(masks_dir, mask_file),
                    os.path.join(test_masks_dir, mask_file))

    print("Dataset splitting completed.")
