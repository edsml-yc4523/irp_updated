"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import torch


def calculate_recall(outputs, targets, num_classes=4):
    """
    Calculate the recall for each class.

    Args:
        outputs (torch.Tensor): Model outputs (logits or probabilities).
        targets (torch.Tensor): Ground truth labels.
        num_classes (int): Number of classes. Default is 4.

    Returns:
        torch.Tensor: Tensor containing recall for each class.
    """
    device = targets.device

    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(outputs)
    preds = torch.argmax(probabilities, dim=1)

    recalls = []
    for c in range(num_classes):
        true_positive = ((preds == c) & (targets == c)).to(device)
        false_negative = ((preds != c) & (targets == c)).to(device)

        TP = true_positive.sum().float().to(device)
        FN = false_negative.sum().float().to(device)

        if TP + FN == 0:
            recall = torch.tensor(0., device=device)
        else:
            recall = TP / (TP + FN)

        recalls.append(recall)

    recalls = torch.stack(recalls).to(device)
    return recalls


def calculate_miou(preds, labels, num_classes=4):
    """
    Calculate the mean Intersection over Union (mIoU).

    Args:
        preds (torch.Tensor):
            Model predictions, shape (batch_size, num_classes, height, width).
        labels (torch.Tensor):
            Ground truth labels, shape (batch_size, height, width).
        num_classes (int): Number of classes. Default is 4.

    Returns:
        float: Mean IoU across all classes.
    """
    preds = torch.argmax(preds, dim=1)
    iou_per_class = []

    for cls in range(num_classes):
        pred_mask = (preds == cls)
        label_mask = (labels == cls)

        intersection = (pred_mask & label_mask).sum().item()
        union = (pred_mask | label_mask).sum().item()

        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union

        iou_per_class.append(iou)

    mean_iou = torch.tensor(
        [iou for iou in iou_per_class if not torch.isnan(torch.tensor(iou))]
        ).mean().item()

    return mean_iou
