"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import torch
from sentinel_segmentation.loss_function.losses import (DiceLoss,
                                                        FocalLoss,
                                                        CombinedLoss)


def test_dice_loss():
    loss_fn = DiceLoss()
    inputs = torch.randn(2, 4, 256, 256)
    targets = torch.randint(0, 4, (2, 256, 256))
    loss = loss_fn(inputs, targets)
    assert loss.item() >= 0, "Loss should be non-negative."


def test_focal_loss():
    loss_fn = FocalLoss()
    inputs = torch.randn(2, 4, 256, 256)
    targets = torch.randint(0, 4, (2, 256, 256))
    loss = loss_fn(inputs, targets)
    assert loss.item() >= 0, "Loss should be non-negative."


def test_combined_loss():
    loss_fn = CombinedLoss()
    inputs = torch.randn(2, 4, 256, 256)
    targets = torch.randint(0, 4, (2, 256, 256))
    loss = loss_fn(inputs, targets)
    assert loss.item() >= 0, "Loss should be non-negative."
