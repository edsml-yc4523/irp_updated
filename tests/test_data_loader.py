"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import os
import pytest
import numpy as np
from PIL import Image
from sentinel_segmentation.data_loader.dataset import (SegmentationDataset,
                                                       compute_mean_std)


@pytest.fixture
def dummy_data_dir(tmpdir):
    data_dir = tmpdir.mkdir("data")
    for i in range(5):
        image = np.random.randint(0, 256, (128, 128, 6), dtype=np.uint8)
        image_path = os.path.join(data_dir, f"image_{i}.npy")
        np.save(image_path, image)
        mask = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        mask_image = Image.fromarray(mask)
        mask_image.save(os.path.join(data_dir, f"image_{i}.png"))
    return str(data_dir)


def test_compute_mean_std(dummy_data_dir):
    mean, std = compute_mean_std(dummy_data_dir)
    assert mean.shape == (6,), f"Unexpected mean shape: {mean.shape}"
    assert std.shape == (6,), f"Unexpected std shape: {std.shape}"


def test_segmentation_dataset(dummy_data_dir):
    dummy_masks_dir = dummy_data_dir
    color_to_label = {
        (0, 0, 0): 0, (255, 0, 0): 1,
        (0, 255, 0): 2, (0, 0, 255): 3
    }
    dataset = SegmentationDataset(
        dummy_data_dir, dummy_masks_dir, (128, 128),
        color_to_label, mean=np.zeros(6), std=np.ones(6)
    )
    assert len(dataset) == 10, f"Unexpected dataset length: {len(dataset)}"
    sample_image, sample_mask = dataset[0]
    assert (
        (
            sample_image.shape
            == (6, 128, 128)
        )
    ), f"Unexpected sample image shape: {sample_image.shape}"
    assert (
        sample_mask.shape == (128, 128)
    ), f"Unexpected sample mask shape: {sample_mask.shape}"
