"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import torch
import pytest
from sentinel_segmentation.models import (
    UnetResnet34WithDropout,
    UnetWithDropout,
    UnetPlusPlusResnet50WithDropout,
    DeepLabV3Plus_resnet50WithDropout
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_input():
    # Creating a dummy input tensor with the shape (batch_size, channels, height, width)
    return torch.randn(1, 6, 128, 128)


def test_unet_resnet34_with_dropout(dummy_input, device):
    model = UnetResnet34WithDropout(dropout_p=0.5).to(device)
    output = model(dummy_input.to(device))
    assert output.shape == (1, 4, 128, 128), (
        f"Unexpected output shape: {output.shape}"
    )


def test_unet_resnet50_with_dropout(dummy_input, device):
    model = UnetWithDropout(dropout_p=0.5).to(device)
    output = model(dummy_input.to(device))
    assert output.shape == (1, 4, 128, 128), (
        f"Unexpected output shape: {output.shape}"
    )


def test_unetplusplus_resnet50_with_dropout(dummy_input, device):
    model = UnetPlusPlusResnet50WithDropout(dropout_p=0.5).to(device)
    output = model(dummy_input.to(device))
    assert output.shape == (1, 4, 128, 128), (
        f"Unexpected output shape: {output.shape}"
    )


def test_deeplabv3plus_resnet50_with_dropout(dummy_input, device):
    model = DeepLabV3Plus_resnet50WithDropout(dropout_p=0.5).to(device)
    model.eval()
    output = model(dummy_input.to(device))
    assert output.shape == torch.Size([1, 4, 128, 128]), f"Unexpected output shape: {output.shape}"
