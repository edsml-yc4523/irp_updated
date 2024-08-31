"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import torch
from sklearn.metrics import confusion_matrix


def evaluate_model(model, test_loader, num_classes, device):
    """
    Evaluate the model on the test dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        num_classes (int): Number of classes.
        device (torch.device): Device to use for evaluation.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    iou_per_class = []
    dice_per_class = []
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    confusion_mat = torch.zeros(num_classes, num_classes)

    for cls in range(num_classes):
        pred_cls = all_preds == cls
        target_cls = all_targets == cls

        intersection = (pred_cls & target_cls).float().sum()
        union = (pred_cls | target_cls).float().sum()

        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_per_class.append(iou)

        dice = (2 * intersection + 1e-6) / (
            pred_cls.float().sum() + target_cls.float().sum() + 1e-6)
        dice_per_class.append(dice)

        precision = (intersection + 1e-6) / (pred_cls.float().sum() + 1e-6)
        recall = (intersection + 1e-6) / (target_cls.float().sum() + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)

        print(
            f'Class {cls}: IoU = {iou:.4f}, Dice = {dice:.4f},'
            f'Precision = {precision:.4f},'
            f'Recall = {recall:.4f}, F1 Score = {f1:.4f}'
            )

    mean_iou = sum(iou_per_class) / num_classes
    mean_dice = sum(dice_per_class) / num_classes
    mean_precision = sum(precision_per_class) / num_classes
    mean_recall = sum(recall_per_class) / num_classes
    mean_f1 = sum(f1_per_class) / num_classes

    correct = (all_preds == all_targets).float().sum()
    accuracy = correct / all_targets.numel()

    confusion_mat += confusion_matrix(
        all_targets.view(-1), all_preds.view(-1),
        labels=list(range(num_classes))
    )

    class_accuracies = confusion_mat.diag() / confusion_mat.sum(1)
    balanced_accuracy = class_accuracies.mean().item()

    print(f'\nTest Mean IoU: {mean_iou:.4f}, Test Accuracy: {accuracy:.4f}, '
          f'Test Mean Dice: {mean_dice:.4f}')
    print(f'Test Mean Precision: {mean_precision:.4f}, '
          f'Test Mean Recall: {mean_recall:.4f}, '
          f'Test Mean F1 Score: {mean_f1:.4f}')
    print(f'Balanced Accuracy: {balanced_accuracy:.4f}')

    return {
        'iou': mean_iou.item(),
        'accuracy': accuracy.item(),
        'dice': mean_dice.item(),
        'precision': mean_precision.item(),
        'recall': mean_recall.item(),
        'f1': mean_f1.item(),
        'balanced_accuracy': balanced_accuracy,
        'confusion_matrix': confusion_mat
    }
