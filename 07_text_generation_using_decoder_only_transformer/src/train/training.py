import torch
import torchmetrics

from src.utils.utils import accuracy_fn


def train_step(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device
):
    train_loss = 0

    # Putting the model to training mode
    model.train()

    for batch, (x, y) in enumerate(train_dataloader):
        # (batch, seq_len)
        x = x.to(device)
        y = y.to(device)

        # 1-D array of all the tokens/words
        # (batch * seq_len)
        y = y.view(y.size(0) * y.size(1))

        # Forward pass
        # (batch, seq_len, vocab_size)
        encoder_output = model.encode(x)
        y_logits = model.project(encoder_output)

        # (batch * seq_len, vocab_size)
        y_logits = y_logits.view(y_logits.size(0) * y_logits.size(1), y_logits.size(2))

        # Calculate the loss for the batch and add it
        loss = loss_fn(y_logits, y)
        train_loss += loss

        # Clear out existing gradients
        optimizer.zero_grad()

        # Backpropagation of loss w.r.t. all the model parameters (Calculates the gradients)
        loss.backward()

        # Calculates step size and updates model parameters
        optimizer.step()

    train_loss = train_loss / len(train_dataloader)

    return train_loss


def train_model(
        epochs: int,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device
):

    for epoch in torch.arange(epochs):
        train_loss = train_step(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            device
        )

        print(f'Epoch {epoch + 1}, Train Loss {train_loss}')
