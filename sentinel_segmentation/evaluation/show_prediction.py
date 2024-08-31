"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sentinel_segmentation.visualization.utils import label_to_color, denormalize_image


def show_predictions(model, test_loader, device,
                     color_map, mean, std, num_images=5):
    """
    Display predictions of a model alongside
    the original images and true masks.

    Args:
        model (torch.nn.Module): The model to use for prediction.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        device (torch.device): Device to run the model on.
        color_map (dict): Mapping of colors to labels.
        mean (list or np.ndarray): Mean values used for normalization.
        std (list or np.ndarray): Standard deviation values
                                    used for normalization.
        num_images (int, optional): Number of images to display.
                                    Defaults to 27.
    """
    model.eval()
    images_handled = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            if images_handled >= num_images:
                break

            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            inputs = inputs.cpu().numpy().transpose(0, 2, 3, 1)
            targets = targets.cpu().numpy()
            preds = preds.cpu().numpy()

            for i in range(inputs.shape[0]):
                if images_handled >= num_images:
                    break

                plt.figure(figsize=(10, 6))

                original_img = denormalize_image(inputs[i], mean, std)

                if original_img.shape[-1] >= 3:
                    original_img_rgb = original_img[:, :, :3]
                else:
                    original_img_rgb = original_img

                plt.subplot(1, 3, 1)
                plt.imshow(np.clip(original_img_rgb / 255.0, 0, 1))
                plt.title('Original Image')
                plt.axis('off')

                true_color_mask = label_to_color(targets[i], color_map)
                plt.subplot(1, 3, 2)
                plt.imshow(true_color_mask)
                plt.title('True Mask')
                plt.axis('off')

                pred_color_mask = label_to_color(preds[i], color_map)
                plt.subplot(1, 3, 3)
                plt.imshow(pred_color_mask)
                plt.title('Predicted Mask')
                plt.axis('off')

                plt.show()
                images_handled += 1


def predict_single_image(model, image, device):
    """
    Predict the class of a single image using a pre-trained model.
    Parameters:
    ----------
    model : torch.nn.Module
        The pre-trained deep learning model used for prediction.

    image : torch.Tensor
        The input image as a tensor,
        typically a 3D tensor with shape (C, H, W).

    device : torch.device
        The device (e.g., 'cuda' or 'cpu') on which the model
        and image should be processed.

    Returns:
    ----------
    numpy.ndarray
        The predicted class index as a NumPy array.
    """
    model.eval()
    with torch.no_grad():
        image = image.to(device, dtype=torch.float)
        output = model(image)
        prediction = torch.argmax(output, dim=1)
    return prediction.cpu().numpy().squeeze(0)


def show_single_prediction(image, prediction, color_map, mean, std):
    """
    Display the original image and its predicted mask side by side.

    Args:
        image (torch.Tensor): The input image tensor.
        prediction (np.ndarray): The predicted mask.
        color_map (dict): A dictionary mapping labels to colors.
        mean (list): The mean values used for normalization.
        std (list): The standard deviation values used for normalization.
    """
    plt.figure(figsize=(10, 6))

    # Adjust the image tensor to match the expected shape (C, H, W)
    if image.ndim == 4:  # If the image has a batch dimension, remove it
        image = image.squeeze(0)

    # Denormalize and convert the image back to its original form
    original_img = denormalize_image(
        image.numpy().transpose(1, 2, 0), mean, std
        )

    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(np.clip(original_img[:, :, :3] / 255.0, 0, 1))
    plt.title('Original Image')
    plt.axis('off')

    # Convert the predicted labels to a color mask
    pred_color_mask = label_to_color(prediction, color_map)

    # Display the predicted mask
    plt.subplot(1, 2, 2)
    plt.imshow(pred_color_mask)
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.show()
