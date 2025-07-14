import torch
from torchmetrics import Accuracy


# Returns a device to be used for training/inference
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


# Returns an accuracy function
def accuracy_fn(num_classes: int):
    return Accuracy(task='multiclass', num_classes=num_classes)
