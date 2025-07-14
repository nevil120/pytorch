import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Create dataset and dataloader
def create_dataloaders(train_dir, 
                       val_dir, 
                       train_transform,
                       val_transform,
                       batch_size):
    
    # Create train, val dataset from custom images
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    class_names = train_dataset.classes
    
    # Create train, val dataloader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader, class_names
