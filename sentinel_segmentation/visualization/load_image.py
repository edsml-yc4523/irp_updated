"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import torch
import numpy as np
from torchvision import transforms


def load_and_preprocess_image(image_path, img_size, mean, std):
    """
    Load and preprocess an image for model prediction.

    Args:
        image_path (str): Path to the image file.
        img_size (tuple): Desired size for the image (H, W).
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.

    Returns:
        torch.Tensor: Preprocessed image ready for model input.
    """
    image = np.load(image_path)
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).float()

    # resize = transforms.Resize(img_size)
    # image = resize(image)

    # center_crop = transforms.CenterCrop((116, 116))
    # image = center_crop(image)

    resize = transforms.Resize(img_size)
    image = resize(image)

    image = (image - torch.tensor(mean).view(6, 1, 1)
             ) / torch.tensor(std).view(6, 1, 1)
    image = image.unsqueeze(0)

    return image
