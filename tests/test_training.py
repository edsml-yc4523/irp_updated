"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import torch
import pytest
from torch.utils.data import DataLoader
from sentinel_segmentation.training.train import (
    adjust_learning_rate,
    cosine_lr_scheduler,
    train_model
)


@pytest.fixture
def optimizer():
    model = torch.nn.Linear(10, 2)
    return torch.optim.Adam(model.parameters(), lr=0.001)


def test_adjust_learning_rate():
    init_lr = 1e-4
    min_lr = init_lr * 0.01
    batch_size = 16
    adjusted_lr, adjusted_min_lr = adjust_learning_rate(
        batch_size, init_lr, min_lr
    )
    assert adjusted_lr > 0, "Adjusted learning rate should be positive."
    assert adjusted_min_lr > 0, "Adjusted min learning rate should be positive."


def test_cosine_lr_scheduler(optimizer):
    init_lr = 1e-4
    min_lr = init_lr * 0.01
    total_iters = 120
    warmup_total_iters = 3
    warmup_lr_start = 1e-6
    no_aug_iter = 15
    current_iter = 60
    scheduler = cosine_lr_scheduler(
        init_lr, min_lr, total_iters, warmup_total_iters,
        warmup_lr_start, no_aug_iter, current_iter
    )
    assert scheduler > 0, "Scheduler learning rate should be positive."


def test_train_model():
    model = torch.nn.Conv2d(3, 4, 3, padding=1)
    dataset = torch.utils.data.TensorDataset(
        torch.randn(10, 3, 128, 128), torch.randint(0, 4, (10, 128, 128)))
    loader = DataLoader(dataset, batch_size=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_losses, _, _, _, _ = train_model(
        model, loader, loader, criterion, optimizer,
        torch.device('cpu'), num_epochs=2
    )
    assert len(train_losses) == 2, "Should return a loss for each epoch."
