"""
Author: Yibing Chen
GitHub username: edsml-yc4523
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from sentinel_segmentation.evaluation.metrics import (
    calculate_miou, calculate_recall
)
from sentinel_segmentation.evaluation.evaluate import evaluate_model


@pytest.fixture
def dummy_predictions():
    return torch.randn(1, 4, 128, 128)


@pytest.fixture
def dummy_targets():
    return torch.randint(0, 4, (1, 128, 128))


def test_calculate_miou(dummy_predictions, dummy_targets):
    miou = calculate_miou(dummy_predictions, dummy_targets, num_classes=4)
    assert 0 <= miou <= 1, f"mIoU should be between 0 and 1, got {miou}"


def test_calculate_recall(dummy_predictions, dummy_targets):
    recall = calculate_recall(dummy_predictions, dummy_targets)
    assert recall.shape == (4,), f"Recall should have shape (4,), got {recall.shape}"
    assert (recall >= 0).all() and (recall <= 1).all(), (
        "Recall values should be between 0 and 1"
    )


class DummyModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(DummyModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def test_evaluate_model():
    # Set up the dummy model, data, and other parameters
    num_classes = 4
    model = DummyModel(num_classes)
    device = torch.device('cpu')
    
    # Create dummy data
    inputs = torch.randn(10, 3, 128, 128)
    targets = torch.randint(0, num_classes, (10, 128, 128))
    
    # Create a DataLoader
    dataset = TensorDataset(inputs, targets)
    data_loader = DataLoader(dataset, batch_size=2)
    
    # Evaluate the model
    metrics = evaluate_model(model, data_loader, num_classes, device)
    
    # Assertions to check that the metrics are as expected
    assert isinstance(metrics, dict), "Metrics should be returned as a dictionary"
    
    expected_keys = [
        'iou', 'accuracy', 'dice', 
        'precision', 'recall', 'f1', 
        'balanced_accuracy', 'confusion_matrix'
    ]
    
    for key in expected_keys:
        assert key in metrics, f"Metrics dictionary should contain key '{key}'"
    
    # Check that IoU, accuracy, dice, precision, recall, and F1 score are within expected ranges
    assert 0 <= metrics['iou'] <= 1, "IoU should be between 0 and 1"
    assert 0 <= metrics['accuracy'] <= 1, "Accuracy should be between 0 and 1"
    assert 0 <= metrics['dice'] <= 1, "Dice should be between 0 and 1"
    assert 0 <= metrics['precision'] <= 1, "Precision should be between 0 and 1"
    assert 0 <= metrics['recall'] <= 1, "Recall should be between 0 and 1"
    assert 0 <= metrics['f1'] <= 1, "F1 score should be between 0 and 1"
    assert metrics['confusion_matrix'].shape == (num_classes, num_classes), \
        f"Confusion matrix should have shape ({num_classes}, {num_classes})"
    
    print("All assertions passed.")