import torch
from torchmetrics import Accuracy

from pathlib import Path


# Setup target device
def set_device_agnostic_mode():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    return device


# Accuracy function for multi-class classification
def accuracy_fn(num_classes: int):
    return Accuracy(task='multiclass', num_classes=num_classes)


# Save trained model
def save_model(model,
               target_dir,
               model_name):
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)
