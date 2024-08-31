"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import numpy as np


def label_to_color(label_mask, color_map):
    """
    Convert label mask to color mask using the specified color map.

    Args:
        label_mask (np.ndarray): The label mask array.
        color_map (dict): A dictionary mapping colors to labels.

    Returns:
        np.ndarray: The color mask array.
    """
    color_mask = np.zeros((label_mask.shape[0], label_mask.shape[1], 3),
                          dtype=np.uint8)
    for key, value in color_map.items():
        color_mask[label_mask == value] = np.array(key)
    return color_mask


def denormalize_image(img, mean, std):
    """
    Denormalize the image using the provided mean and standard deviation.

    Args:
        img (np.ndarray): The normalized image.
        mean (list): The mean values used for normalization.
        std (list): The standard deviation values used for normalization.

    Returns:
        np.ndarray: The denormalized image.
    """
    mean = np.array(mean).reshape(1, 1, -1)
    std = np.array(std).reshape(1, 1, -1)
    img_denormalized = img * std + mean
    img_denormalized = np.clip(img_denormalized, 0, 255)
    return img_denormalized
