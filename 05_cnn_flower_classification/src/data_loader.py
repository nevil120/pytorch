from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def create_dataloaders(train_dir: str,
                       test_dir: str,
                       train_transforms: transforms.Compose,
                       test_transforms: transforms.Compose,
                       batch_size: int):
    # Get training data
    train_dataset = datasets.Flowers102(
        root=train_dir,
        download=True,
        split='train',
        transform=train_transforms
    )

    # Get testing data
    test_dataset = datasets.Flowers102(
        root=test_dir,
        download=True,
        split='test',
        transform=test_transforms
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    class_names = train_dataset.classes

    return train_dataloader, test_dataloader, class_names
