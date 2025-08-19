import torch


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
        # Put tensors on the device
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        src_mask = (x == 0).view(x.size(0), 1, 1, x.size(-1))
        encoder_output = model.encode(x, src_mask)

        y_input = y[:, :-1]
        y_output = y[:, 1:]

        decoder_output = model.decode(y_input, encoder_output, src_mask)
        y_logits = model.project(decoder_output)

        # (batch * seq_len, vocab_size)
        y_logits = y_logits.view(y_logits.size(0) * y_logits.size(1), y_logits.size(2))
        # (batch * seq_len)
        y_output = y_output.reshape(-1)

        # Calculate the loss for the batch and add it
        loss = loss_fn(y_logits, y_output)
        train_loss += loss

        # Clear out existing gradients
        optimizer.zero_grad()

        # Backpropagation of loss w.r.t. all the model parameters (Calculates the gradients)
        loss.backward()

        # Calculates step size and updates model parameters
        optimizer.step()

        if (batch % 50) == 0:
            print(f'Train Batch - {batch}, Loss - {loss}')

    train_loss = train_loss / len(train_dataloader)

    return train_loss


def validation_step(
        model: torch.nn.Module,
        val_dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.CrossEntropyLoss,
        device: torch.device
):
    val_loss = 0

    # Putting the model to training mode
    model.eval()

    with torch.inference_mode():
        for batch, (x, y) in enumerate(val_dataloader):
            # Put tensors on the device
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            src_mask = (x == 0).view(x.size(0), 1, 1, x.size(-1))
            encoder_output = model.encode(x, src_mask)

            y_input = y[:, :-1]
            y_output = y[:, 1:]

            decoder_output = model.decode(y_input, encoder_output, src_mask)
            y_logits = model.project(decoder_output)

            # (batch * seq_len, vocab_size)
            y_logits = y_logits.view(y_logits.size(0) * y_logits.size(1), y_logits.size(2))
            # (batch * seq_len)
            y_output = y_output.reshape(-1)

            # Calculate the loss for the batch and add it
            loss = loss_fn(y_logits, y_output)
            val_loss += loss

            if (batch % 50) == 0:
                print(f'Validation Batch - {batch}, Loss - {loss}')

        val_loss = val_loss / len(val_dataloader)

    return val_loss


def train_model(
        epochs: int,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
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

        val_loss = validation_step(
            model,
            val_dataloader,
            loss_fn,
            device
        )

        print(f'Epoch - {epoch + 1}, Train Loss - {train_loss}, Validation Loss - {val_loss}')
