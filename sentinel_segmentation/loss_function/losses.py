"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, cls_weights=None):
        """
        Initializes the Dice Loss function.

        Args:
            smooth (float): Smoothing factor to avoid division by zero.
            cls_weights (torch.Tensor, optional):
            Class weights for weighted loss.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.cls_weights = cls_weights

    def forward(self, inputs, targets):
        """
        Calculates the Dice loss.

        Args:
            inputs (torch.Tensor): Predicted probabilities.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Dice loss value.
        """
        inputs = torch.softmax(inputs, dim=1)
        dice_loss = 0.0
        num_classes = inputs.shape[1]

        for c in range(num_classes):
            input_flat = inputs[:, c].contiguous().view(-1)
            target_flat = (targets == c).float().contiguous().view(-1)
            intersection = torch.sum(input_flat * target_flat)
            union = torch.sum(input_flat) + torch.sum(target_flat)
            dice_score = (2. * intersection + self.smooth
                          ) / (union + self.smooth)

            if self.cls_weights is not None:
                dice_class_loss = (1 - dice_score) * self.cls_weights[c]
            else:
                dice_class_loss = 1 - dice_score

            dice_loss += dice_class_loss

        return dice_loss / num_classes


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.4, gamma=2.0, cls_weights=None):
        """
        Initializes the Focal Loss function.

        Args:
            alpha (float): Balancing factor for class imbalance.
            gamma (float): Focusing parameter to reduce easy examples' loss.
            cls_weights (torch.Tensor, optional):
            Class weights for weighted loss.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cls_weights = cls_weights

    def forward(self, inputs, targets):
        """
        Calculates the Focal loss.

        Args:
            inputs (torch.Tensor): Predicted probabilities.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Focal loss value.
        """
        inputs = torch.softmax(inputs, dim=1)
        focal_loss = 0.0
        num_classes = inputs.shape[1]

        for c in range(num_classes):
            input_flat = inputs[:, c].contiguous().view(-1)
            target_flat = (targets == c).float().contiguous().view(-1)
            logpt = torch.log(input_flat + 1e-6)
            pt = torch.exp(logpt)

            if self.cls_weights is not None:
                logpt = logpt * self.cls_weights[c]

            focal_loss += -((1 - pt) ** self.gamma) * logpt * target_flat

        if self.alpha is not None:
            focal_loss *= self.alpha

        return focal_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_focal=0.5, dice_smooth=1.0,
                 focal_alpha=0.4, focal_gamma=2.0, cls_weights=None,
                 use_dice_weight=True, use_focal_weight=True, use_focal=True):
        """
        Initializes the combined Dice and Focal Loss function.

        Args:
            weight_dice (float): Weight for the Dice loss.
            weight_focal (float): Weight for the Focal loss.
            dice_smooth (float): Smoothing factor for Dice loss.
            focal_alpha (float): Alpha value for Focal loss.
            focal_gamma (float): Gamma value for Focal loss.
            cls_weights (torch.Tensor, optional):
                                        Class weights for weighted loss.
            use_dice_weight (bool): Whether to use class weights for Dice loss.
            use_focal_weight (bool):
                                Whether to use class weights for Focal loss.
            use_focal (bool): Whether to use Focal loss
                                        instead of CrossEntropyLoss.
        """
        super(CombinedLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.use_focal = use_focal

        dice_cls_weights = cls_weights if use_dice_weight else None
        self.dice_loss = DiceLoss(smooth=dice_smooth,
                                  cls_weights=dice_cls_weights)

        if self.use_focal:
            focal_cls_weights = cls_weights if use_focal_weight else None
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma,
                                        cls_weights=focal_cls_weights)
        else:
            ce_cls_weights = cls_weights if use_focal_weight else None
            self.ce_loss = nn.CrossEntropyLoss(weight=ce_cls_weights)

    def forward(self, inputs, targets):
        """
        Calculates the combined loss.

        Args:
            inputs (torch.Tensor): Predicted probabilities.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Combined loss value.
        """
        dice_loss = self.dice_loss(inputs, targets)

        if self.use_focal:
            focal_loss = self.focal_loss(inputs, targets)
            return (self.weight_dice * dice_loss + self.weight_focal *
                    focal_loss)
        else:
            ce_loss = self.ce_loss(inputs, targets)
            return self.weight_dice * dice_loss + self.weight_focal * ce_loss
