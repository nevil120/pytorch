import torch
import torchmetrics

from src.utils.utils import accuracy_fn


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

    for batch, (x, y) in enumerate(train_dataloader):
        # Put tensors on the device
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        mask = (x == 0).view(x.size(0), 1, 1, x.size(-1))
        encoder_output = model.encode(x, mask)
        y_logits = model.project(encoder_output)

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


def test_step(
        model: torch.nn.Module,
        test_dataloader: torch.utils.data.DataLoader,
        accuracy: torchmetrics.Accuracy,
        loss_fn: torch.nn.CrossEntropyLoss,
        device: torch.device
):
    test_loss = 0
    test_acc = 0

    # Putting the model to training mode
    model.eval()

    with torch.inference_mode():
        for batch, (x, y) in enumerate(test_dataloader):
            # Put tensors on the device
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            mask = (x == 0).view(x.size(0), 1, 1, x.size(-1))
            encoder_output = model.encode(x, mask)
            y_logits = model.project(encoder_output)

            # Calculate the loss for the batch and add it
            loss = loss_fn(y_logits, y)
            test_loss += loss

            # Calculate the accuracy for the batch and add it
            acc = accuracy(torch.argmax(y_logits, dim=1), y)
            test_acc += acc

        test_loss = test_loss / len(test_dataloader)
        test_acc = test_acc / len(test_dataloader)

    return test_loss, test_acc


def train_model(
        epochs: int,
        num_classes: int,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
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

        test_loss, test_acc = test_step(
            model,
            test_dataloader,
            accuracy,
            loss_fn,
            device
        )

        print(f'Epoch {epoch + 1}, Train Loss {train_loss}, Train Accuracy {train_acc}, Test Loss {test_loss}, '
              f'Test Accuracy {test_acc}')
