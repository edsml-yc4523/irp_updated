"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import os
from PIL import Image


def resize_images(directory, size=(128, 128)):
    """
    Resize all images in the specified directory to the given size.

    Args:
        directory (str): Path to the directory containing the images.
        size (tuple of int, int): Desired output size (width, height).

    """
    for filename in os.listdir(directory):
        if filename.lower().endswith((
            '.jpg', '.jpeg', '.png', '.bmp', '.gif'
        )):
            img_path = os.path.join(directory, filename)
            with Image.open(img_path) as img:
                img = img.resize(size, Image.LANCZOS)
                img.save(img_path)
                print(f"Resized and saved {filename}")
