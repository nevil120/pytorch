import torch
from torchmetrics import Accuracy


def set_device_agnostic_mode():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    return device


def accuracy_fn(num_classes: int):
    return Accuracy(task='multiclass', num_classes=num_classes)
