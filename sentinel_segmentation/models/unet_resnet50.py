"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import torch.nn as nn
import segmentation_models_pytorch as smp


class UnetWithDropout(nn.Module):
    def __init__(self, dropout_p=0.5):
        """
        UNet model with ResNet50 backbone and dropout.

        Args:
            dropout_p (float): Probability of dropout.
        """
        super(UnetWithDropout, self).__init__()
        self.model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=6,
            classes=4,
            decoder_attention_type="scse"
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
