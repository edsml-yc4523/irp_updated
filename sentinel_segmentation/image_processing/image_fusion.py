"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import os
import numpy as np
from PIL import Image


def fuse_images_masks(base_dirs, output_images_dir, output_masks_dir):
    """
    Fuse RGB, NIR, NDWI, and NDVI images into a multi-channel image
    and save as .npy files.

    Args:
        base_dirs (dict): Dictionary containing paths for
                            'RGB', 'NIR', 'NDWI', and 'NDVI' images.
        output_images_dir (str): Directory to save fused images.
        output_masks_dir (str): Directory to save corresponding masks.
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    image_names = [
        f for f in os.listdir(os.path.join(base_dirs['RGB'], 'images'))
        if f.endswith(('.jpg', '.png', '.tif'))
    ]

    counter = 1

    for image_name in image_names:
        base_name = os.path.splitext(image_name)[0]

        # Load RGB image
        rgb_image = Image.open(os.path.join(base_dirs['RGB'], 'images', image_name))
        rgb_array = np.array(rgb_image)

        # Load NIR image
        nir_filename = base_name.replace('rgb', 'NIR').upper() + '.jpg'
        nir_image_path = os.path.join(base_dirs['NIR'], 'images', nir_filename)
        if not os.path.exists(nir_image_path):
            print(f"File not found: {nir_image_path}. Skipping this set.")
            continue
        nir_image = Image.open(nir_image_path)
        nir_array = np.array(nir_image)

        # Load NDWI image
        ndwi_filename = base_name.replace('rgb', 'NDWI').upper() + '.jpg'
        ndwi_image_path = os.path.join(base_dirs['NDWI'], 'images', ndwi_filename)
        if not os.path.exists(ndwi_image_path):
            print(f"File not found: {ndwi_image_path}. Skipping this set.")
            continue
        ndwi_image = Image.open(ndwi_image_path)
        ndwi_array = np.array(ndwi_image)

        # Load NDVI image
        ndvi_filename = base_name.replace('rgb', 'NDVI').upper() + '.jpg'
        ndvi_image_path = os.path.join(base_dirs['NDVI'], 'images', ndvi_filename)
        if not os.path.exists(ndvi_image_path):
            print(f"File not found: {ndvi_image_path}. Skipping this set.")
            continue
        ndvi_image = Image.open(ndvi_image_path)
        ndvi_array = np.array(ndvi_image)

        # Check if dimensions match
        if rgb_array.shape[:2] != nir_array.shape[:2] or \
           rgb_array.shape[:2] != ndwi_array.shape[:2] or \
           rgb_array.shape[:2] != ndvi_array.shape[:2]:
            print(f"Size mismatch between images for {base_name}. Skipping this set.")
            continue

        # Combine the images into a multi-channel image
        nir_array = nir_array[:, :, np.newaxis] if nir_array.ndim == 2 else nir_array
        ndwi_array = ndwi_array[:, :, np.newaxis] if ndwi_array.ndim == 2 else ndwi_array
        ndvi_array = ndvi_array[:, :, np.newaxis] if ndvi_array.ndim == 2 else ndvi_array
        combined_image = np.concatenate([rgb_array, nir_array, ndwi_array, ndvi_array], axis=2)

        # Save the combined image as a .npy file
        output_image_path = os.path.join(output_images_dir, f'combined_{counter}.npy')
        np.save(output_image_path, combined_image)

        # Copy and save the mask file
        mask_file = os.path.join(base_dirs['RGB'], 'masks', base_name + '.png')
        output_mask_file = os.path.join(output_masks_dir, f'combined_{counter}.png')
        if os.path.exists(mask_file):
            mask_image = Image.open(mask_file)
            mask_image.save(output_mask_file)
        else:
            print(f"Mask file not found: {mask_file}. Skipping this set.")

        counter += 1

    print("Image fusion and saving, and mask copying completed.")


def load_image_as_array(image_path):
    """
    Load an image and convert it to a NumPy array.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: The image as a NumPy array.
    """
    image = Image.open(image_path)
    return np.array(image)


def fuse_image(rgb_path, nir_path, ndwi_path, ndvi_path, output_path):
    """
    Fuse RGB, NIR, NDWI, and NDVI images into a multi-channel image
    and save as a .npy file.

    Args:
        rgb_path (str): Path to the RGB image.
        nir_path (str): Path to the NIR image.
        ndwi_path (str): Path to the NDWI image.
        ndvi_path (str): Path to the NDVI image.
        output_path (str): Path to save the fused .npy file.

    Returns:
        None
    """
    # Load each image as a NumPy array
    rgb_array = load_image_as_array(rgb_path)
    nir_array = load_image_as_array(nir_path)
    ndwi_array = load_image_as_array(ndwi_path)
    ndvi_array = load_image_as_array(ndvi_path)

    # Ensure all images have the same dimensions
    if not (rgb_array.shape[:2] == nir_array.shape[:2] == 
            ndwi_array.shape[:2] == ndvi_array.shape[:2]):
        raise ValueError("All images must have the same dimensions")

    # Convert grayscale images to 3D by adding a channel dimension if necessary
    nir_array = nir_array[:, :, np.newaxis] if nir_array.ndim == 2 else nir_array
    ndwi_array = ndwi_array[:, :, np.newaxis] if ndwi_array.ndim == 2 else ndwi_array
    ndvi_array = ndvi_array[:, :, np.newaxis] if ndvi_array.ndim == 2 else ndvi_array

    # Concatenate all the images along the channel axis
    combined_image = np.concatenate([rgb_array, nir_array, ndwi_array, ndvi_array], axis=2)

    # Save the combined image as a .npy file
    np.save(output_path, combined_image)
    print(f"Fused image saved at {output_path}")
