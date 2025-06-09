import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloaders(train_dir, 
                       test_dir, 
                       train_transform,
                       test_transform,
                       batch_size):
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    
    class_names = train_dataset.classes
    
    train_data_loader = DataLoader(
        train_dataset, 
        batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    test_data_loader = DataLoader(
        test_dataset, 
        batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    return train_data_loader, test_data_loader, class_names
