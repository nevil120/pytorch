import torch
import torchmetrics

from src.utils import accuracy_fn


def train_step(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        accuracy: torchmetrics.Accuracy,
        loss_fn: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device
):
    train_loss = 0
    train_acc = 0

    # Putting the model to training mode
    model.train()

    # Iterate through training data
    for batch, (x, y) in enumerate(train_dataloader):
        # Put tensors on the device
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        y_logits = model(x)

        # Calculate the loss for the batch and add it
        loss = loss_fn(y_logits, y)
        train_loss += loss

        # Calculate the accuracy for the batch and add it
        acc = accuracy(torch.argmax(y_logits, dim=1), y)
        train_acc += acc

        # Clear out existing gradients
        optimizer.zero_grad()

        # Backpropagation of loss w.r.t. all the model parameters (Calculates the gradients)
        loss.backward()

        # Calculates step size and updates model parameters
        optimizer.step()

    train_loss = train_loss / len(train_dataloader)
    train_acc = train_acc / len(train_dataloader)

    return train_loss, train_acc


def val_step(
        model: torch.nn.Module,
        val_dataloader: torch.utils.data.DataLoader,
        accuracy: torchmetrics.Accuracy,
        loss_fn: torch.nn.CrossEntropyLoss,
        device: torch.device
):
    val_loss = 0
    val_acc = 0

    # Putting the model to training mode
    model.eval()

    with torch.inference_mode():
        # Iterate through validation data
        for batch, (x, y) in enumerate(val_dataloader):
            # Put tensors on the device
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            y_logits = model(x)

            # Calculate the loss for the batch and add it
            loss = loss_fn(y_logits, y)
            val_loss += loss

            # Calculate the accuracy for the batch and add it
            acc = accuracy(torch.argmax(y_logits, dim=1), y)
            val_acc += acc

        val_loss = val_loss / len(val_dataloader)
        val_acc = val_acc / len(val_dataloader)

    return val_loss, val_acc


def train_model(
        epochs: int,
        num_classes: int,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: torch.device
):
    accuracy = accuracy_fn(num_classes).to(device)

    for epoch in torch.arange(epochs):
        train_loss, train_acc = train_step(
            model,
            train_dataloader,
            accuracy,
            loss_fn,
            optimizer,
            device
        )

        val_loss, val_acc = val_step(
            model,
            val_dataloader,
            accuracy,
            loss_fn,
            device
        )

        print(f'Epoch {epoch + 1}, Train Loss {train_loss}, Train Accuracy {train_acc}, Val Loss {val_loss}, '
              f'Val Accuracy {val_acc}')

        scheduler.step()
