"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import torch
import math
from functools import partial
from sentinel_segmentation.evaluation.metrics import (calculate_miou,
                                                      calculate_recall)


def adjust_learning_rate(batch_size, init_lr=1e-4, min_lr=1e-6, nbs=16,
                         lr_limit_min=3e-4, lr_limit_max=5e-4):
    """
    Adjust the learning rate based on the batch size.

    Args:
        batch_size (int): Current batch size.
        init_lr (float): Initial learning rate.
        min_lr (float): Minimum learning rate.
        nbs (int): Nominal batch size used to
                            determine the learning rate scale.
        lr_limit_min (float): Minimum allowed learning rate.
        lr_limit_max (float): Maximum allowed learning rate.

    Returns:
        tuple: Adjusted initial learning rate and minimum learning rate.
    """
    init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min),
                      lr_limit_max)
    min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2),
                     lr_limit_max * 1e-2)
    return init_lr_fit, min_lr_fit


def cosine_lr_scheduler(lr, min_lr, total_iters, warmup_total_iters,
                        warmup_lr_start, no_aug_iter, iters):
    """
    Cosine annealing learning rate scheduler with warmup.

    Args:
        lr (float): Initial learning rate.
        min_lr (float): Minimum learning rate.
        total_iters (int): Total number of iterations.
        warmup_total_iters (int): Number of warmup iterations.
        warmup_lr_start (float): Starting learning rate during warmup.
        no_aug_iter (int): Number of iterations with no augmentation.
        iters (int): Current iteration.

    Returns:
        float: Adjusted learning rate.
    """
    if iters <= warmup_total_iters:
        lr = ((lr - warmup_lr_start) * pow(
            iters / float(warmup_total_iters), 2
            )
              + warmup_lr_start)
    elif iters >= total_iters - no_aug_iter:
        lr = min_lr
    else:
        lr = (min_lr + 0.5 * (lr - min_lr) *
              (1.0 + math.cos(math.pi * (iters - warmup_total_iters)
               / (total_iters - warmup_total_iters - no_aug_iter))))
    return lr


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    """
    Set the learning rate for the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_scheduler_func (function): Learning rate scheduler function.
        epoch (int): Current epoch.
    """
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_freeze_by_stage(model, stage):
    """
    Set the layers to be frozen or unfrozen based on the training stage.

    Args:
        model (torch.nn.Module): The model.
        stage (int): The current stage of training.
    """
    if stage == 1:
        for name, param in model.model.encoder.named_parameters():
            if "layer1" in name or "layer2" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    elif stage == 2:
        for name, param in model.model.encoder.named_parameters():
            if "layer1" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        for param in model.model.encoder.parameters():
            param.requires_grad = True


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs=120, save_path='best_model.pth'):
    """
    Train the model.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader):
                                DataLoader for training data.
        val_loader (torch.utils.data.DataLoader):
                                DataLoader for validation data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to use for training.
        num_epochs (int): Number of epochs to train.
        save_path (str): Path to save the best model.

    Returns:
        tuple: Lists of training losses, validation losses, training mIoUs,
        validation mIoUs, and validation recalls.
    """
    train_losses = []
    val_losses = []
    train_mious = []
    val_mious = []
    val_recalls = []

    stage = 1
    batch_size = 16
    best_val_miou = 0.0

    init_lr_fit, min_lr_fit = adjust_learning_rate(
        batch_size, optimizer.defaults['lr'], optimizer.defaults['lr'] * 0.01)
    lr_scheduler_func = partial(
        cosine_lr_scheduler, init_lr_fit, min_lr_fit, num_epochs,
        3, max(0.1 * init_lr_fit, 1e-6), 15)

    for epoch in range(num_epochs):
        if epoch == 40:
            stage = 2
            set_freeze_by_stage(model, stage)
            batch_size = 8
            init_lr_fit, min_lr_fit = adjust_learning_rate(
                batch_size, optimizer.defaults['lr'],
                optimizer.defaults['lr'] * 0.01)
            lr_scheduler_func = partial(
                cosine_lr_scheduler, init_lr_fit, min_lr_fit, num_epochs,
                3, max(0.1 * init_lr_fit, 1e-6), 15)
        elif epoch == 80:
            stage = 3
            set_freeze_by_stage(model, stage)
            batch_size = 4
            init_lr_fit, min_lr_fit = adjust_learning_rate(
                batch_size, optimizer.defaults['lr'],
                optimizer.defaults['lr'] * 0.01)
            lr_scheduler_func = partial(
                cosine_lr_scheduler, init_lr_fit, min_lr_fit, num_epochs,
                3, max(0.1 * init_lr_fit, 1e-6), 15)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        model.train()
        total_train_loss = 0
        total_train_miou = 0

        for inputs, targets in train_loader:
            inputs, targets = (inputs.to(device, dtype=torch.float),
                               targets.to(device))
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            miou = calculate_miou(outputs, targets)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_miou += miou

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_miou = total_train_miou / len(train_loader)
        train_losses.append(avg_train_loss)
        train_mious.append(avg_train_miou)

        model.eval()
        total_val_loss = 0
        total_val_miou = 0
        total_val_recall = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = (inputs.to(device, dtype=torch.float),
                                   targets.to(device))
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                miou = calculate_miou(outputs, targets)
                recall = calculate_recall(outputs, targets)

                total_val_loss += loss.item()
                total_val_miou += miou
                total_val_recall += recall.mean().item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_miou = total_val_miou / len(val_loader)
        avg_val_recall = total_val_recall / len(val_loader)
        val_losses.append(avg_val_loss)
        val_mious.append(avg_val_miou)
        val_recalls.append(avg_val_recall)

        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            torch.save(model.state_dict(), save_path)
            print(f'Saved best model with mIoU: {best_val_miou:.4f}')

        print(
            f'Epoch {epoch + 1}/{num_epochs}, '
            f'LR: {optimizer.param_groups[0]["lr"]:.6f}, '
            f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
            f'Train mIoU: {avg_train_miou:.4f}, Val mIoU: {avg_val_miou:.4f}, '
            f'Val Recall: {avg_val_recall:.4f}'
        )

    return train_losses, val_losses, train_mious, val_mious, val_recalls
