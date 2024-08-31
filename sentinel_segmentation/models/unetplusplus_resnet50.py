"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import torch.nn as nn
import segmentation_models_pytorch as smp


class UnetPlusPlusResnet50WithDropout(nn.Module):
    def __init__(self, dropout_p=0.5):
        """
        Initializes the UNet++ model with a ResNet50 backbone
        and dropout layers.

        Args:
            dropout_p (float): Dropout probability.
        """
        super(UnetPlusPlusResnet50WithDropout, self).__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=6,
            classes=4,
            decoder_attention_type="scse"
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after segmentation.
        """
        encoder_features = self.model.encoder(x)
        encoder_features = [self.dropout(f) for f in encoder_features]
        x = self.model.decoder(*encoder_features)
        x = self.dropout(x)
        x = self.model.segmentation_head(x)
        x = self.dropout(x)
        return x
