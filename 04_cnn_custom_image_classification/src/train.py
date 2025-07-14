import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
HIDDEN_UNITS = 20
LEARNING_RATE = 0.01

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
val_dir = "data/pizza_steak_sushi/val"

# Setup target device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# Create train transform
train_transform_trivial_augment = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor() 
])

# Create val transform (no data augmentation)
val_transform_trivial_augment = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Create training and validation dataloader
train_dataloader, val_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    val_dir=val_dir,
    train_transform=train_transform_trivial_augment,
    val_transform=val_transform_trivial_augment,
    batch_size=BATCH_SIZE
)

# Create ImageClassificationModel
model = model_builder.ImageClassificationModel(
    input_channels=3,
    hidden_channels=HIDDEN_UNITS,
    output_channels=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model,
             train_dataloader,
             val_dataloader,
             loss_fn,
             optimizer,
             device,
             NUM_EPOCHS)
