"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import torch.nn as nn
import segmentation_models_pytorch as smp


class DeepLabV3Plus_resnet50WithDropout(nn.Module):
    def __init__(self, dropout_p=0.5):
        """
        DeepLabV3+ model with ResNet50 backbone and dropout.

        Args:
            dropout_p (float): Probability of dropout.
        """
        super(DeepLabV3Plus_resnet50WithDropout, self).__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=6,
            classes=4
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        # Get encoder features and apply dropout
        encoder_features = self.model.encoder(x)
        encoder_features = [self.dropout(f) for f in encoder_features]

        # Decode using the encoder features
        x = self.model.decoder(*encoder_features)
        x = self.dropout(x)

        # Apply dropout to the final output
        x = self.model.segmentation_head(x)
        x = self.dropout(x)

        return x
