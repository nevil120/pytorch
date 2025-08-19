from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


# Create training and validation data loaders
def create_dataloaders(train_dir: str,
                       val_dir: str,
                       train_transforms: transforms.Compose,
                       val_transforms: transforms.Compose,
                       batch_size: int):

    # Get training data
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transforms,
        target_transform=None
    )

    # Get validation data
    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=val_transforms,
        target_transform=None
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, train_dataset.classes
