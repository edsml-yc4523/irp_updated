"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import pytest
import os
import numpy as np
from PIL import Image
from sentinel_segmentation.image_processing.tiff_to_jpg import tif_jpg_transform
from sentinel_segmentation.image_processing.image_resize import resize_images
from sentinel_segmentation.image_processing.image_fusion import fuse_images_masks
from sentinel_segmentation.image_processing.data_split import split_dataset


def create_dummy_tif_file(file_path, size=(128, 128), num_bands=3):
    """
    Creates a dummy TIF file for testing purposes.
    """
    arr = np.random.randint(0, 255, size=(size[1], size[0], num_bands), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(file_path, format="TIFF")


def create_dummy_image_file(file_path, size=(128, 128)):
    """
    Creates a dummy image file for testing purposes.
    """
    arr = np.random.randint(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(file_path)


def test_tif_jpg_transform(tmpdir):
    # Create a dummy TIF file
    tif_path = os.path.join(tmpdir, "test.tif")
    jpg_path = os.path.join(tmpdir, "test.jpg")
    create_dummy_tif_file(tif_path)

    # Test conversion
    tif_jpg_transform(tif_path, jpg_path, band_indices=[0, 1, 2])
    assert os.path.exists(jpg_path), "JPEG file should be created."


def test_resize_images(tmpdir):
    # Create a dummy image
    img_path = os.path.join(tmpdir, "test.jpg")
    create_dummy_image_file(img_path, size=(128, 128))

    # Test resizing
    resize_images(tmpdir, size=(128, 128))
    with Image.open(img_path) as img:
        assert img.size == (128, 128), "Image should be resized to 128x128."


def test_split_dataset(tmpdir):
    # Create a mock dataset
    images_dir = os.path.join(tmpdir, "images")
    masks_dir = os.path.join(tmpdir, "masks")
    os.makedirs(images_dir)
    os.makedirs(masks_dir)

    for i in range(10):
        create_dummy_image_file(os.path.join(images_dir, f"image_{i}.jpg"), size=(128, 128))
        create_dummy_image_file(os.path.join(masks_dir, f"mask_{i}.png"), size=(128, 128))

    # Test dataset split
    output_dir = os.path.join(tmpdir, "split_dataset")
    split_dataset(tmpdir, output_dir)

    train_images_dir = os.path.join(output_dir, "train/images")
    val_images_dir = os.path.join(output_dir, "val/images")
    test_images_dir = os.path.join(output_dir, "test/images")

    assert len(os.listdir(train_images_dir)) > 0, "Train set should not be empty."
    assert len(os.listdir(val_images_dir)) > 0, "Validation set should not be empty."
    assert len(os.listdir(test_images_dir)) > 0, "Test set should not be empty."


def create_dummy_image(directory, filename, size=(128, 128), color=(255, 0, 0), num_channels=3):
    """
    Create a dummy image and save it to the specified directory.
    """
    if num_channels == 1:
        img = Image.new("L", size, color[0])
    else:
        img = Image.new("RGB", size, color)
    img_path = os.path.join(directory, filename)
    img.save(img_path)


@pytest.mark.parametrize("image_size", [(128, 128)])
def test_fuse_images(tmpdir, image_size):
    # Setup directories
    rgb_dir = tmpdir.mkdir("RGB")
    nir_dir = tmpdir.mkdir("NIR")
    ndwi_dir = tmpdir.mkdir("NDWI")
    ndvi_dir = tmpdir.mkdir("NDVI")

    # Create dummy image directories
    base_dirs = {
        'RGB': str(rgb_dir),
        'NIR': str(nir_dir),
        'NDWI': str(ndwi_dir),
        'NDVI': str(ndvi_dir)
    }

    for dir_name in base_dirs.values():
        images_dir = os.path.join(dir_name, 'images')
        masks_dir = os.path.join(dir_name, 'masks')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

    # Create dummy RGB, NIR, NDWI, and NDVI images
    create_dummy_image(os.path.join(base_dirs['RGB'], 'images'), "image_01_rgb.jpg", size=image_size, color=(255, 0, 0), num_channels=3)
    create_dummy_image(os.path.join(base_dirs['NIR'], 'images'), "IMAGE_01_NIR.jpg", size=image_size, color=(0, 255, 0), num_channels=1)
    create_dummy_image(os.path.join(base_dirs['NDWI'], 'images'), "IMAGE_01_NDWI.jpg", size=image_size, color=(0, 0, 255), num_channels=1)
    create_dummy_image(os.path.join(base_dirs['NDVI'], 'images'), "IMAGE_01_NDVI.jpg", size=image_size, color=(255, 255, 0), num_channels=1)

    # Create dummy mask
    create_dummy_image(os.path.join(base_dirs['RGB'], 'masks'), "image_01_rgb.png", size=image_size, color=(128, 128, 128))

    # Create output directories
    output_images_dir = tmpdir.mkdir("output_images")
    output_masks_dir = tmpdir.mkdir("output_masks")

    # Call the fuse_images function
    fuse_images_masks(base_dirs, str(output_images_dir), str(output_masks_dir))

    # Check that the output files are created
    assert len(os.listdir(output_images_dir)) == 1, "One fused image should be created"
    assert len(os.listdir(output_masks_dir)) == 1, "One mask file should be copied"

    # Check the content of the fused image
    fused_image_path = os.path.join(output_images_dir, "combined_1.npy")
    fused_image = np.load(fused_image_path)
    expected_channels = 3 + 1 + 1 + 1  # RGB + NIR + NDWI + NDVI
    assert fused_image.shape == (image_size[1], image_size[0], expected_channels), (
        f"Fused image should have shape {(image_size[1], image_size[0], expected_channels)}, but got {fused_image.shape}"
    )

    # Check the content of the copied mask
    mask_path = os.path.join(output_masks_dir, "combined_1.png")
    assert os.path.exists(mask_path), "Mask file should be copied"
