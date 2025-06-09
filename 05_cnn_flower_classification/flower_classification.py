import torch
import torchvision.models
from torch import nn
from torchinfo import summary
from torchvision import transforms
from torchvision.models import ResNet152_Weights

from src.data_loader import create_dataloaders
from src.models.vgg_16 import VGG16
from src.utils import set_device_agnostic_mode
from src.training import train_model

BATCH_SIZE = 32
COLOR_CHANNELS = 3
EPOCHS = 25

if __name__ == '__main__':

    train_dir = '../data/train'
    test_dir = '../data/test'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataloader, test_dataloader, classes = \
        create_dataloaders(train_dir, test_dir, transform, transform, BATCH_SIZE)

    device = set_device_agnostic_mode()

    vgg_16_model = VGG16(COLOR_CHANNELS, len(classes))
    summary(vgg_16_model, input_size=(BATCH_SIZE, COLOR_CHANNELS, 224, 224))
    vgg_16_model = vgg_16_model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(vgg_16_model.parameters(), lr=0.1)

    train_model(
        EPOCHS,
        len(classes),
        vgg_16_model,
        train_dataloader,
        test_dataloader,
        loss_fn,
        optimizer,
        device
    )

    resnet152_weights = ResNet152_Weights.DEFAULT
    auto_transforms = resnet152_weights.transforms()

    train_dataloader, test_dataloader, classes = \
        create_dataloaders(train_dir, test_dir, auto_transforms, auto_transforms, BATCH_SIZE)

    device = set_device_agnostic_mode()

    resnet152_model = torchvision.models.resnet152(weights=resnet152_weights)

    for name, layer in resnet152_model.named_children():
        if name not in ['fc']:
            for param in layer.parameters():
                param.requires_grad = False

    resnet152_model.fc = nn.Linear(in_features=2048, out_features=len(classes), bias=True)
    summary(resnet152_model,
            input_size=(BATCH_SIZE, COLOR_CHANNELS, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"])
    resnet152_model = resnet152_model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet152_model.parameters(), lr=0.1)

    train_model(
        EPOCHS,
        len(classes),
        resnet152_model,
        train_dataloader,
        test_dataloader,
        loss_fn,
        optimizer,
        device
    )
