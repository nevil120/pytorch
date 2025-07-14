import torch

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc

# Training loop using train dataloader
def train_step(
    model, 
    data_loader,
    loss_fn,
    optimizer,
    device
):

    model.train()
    train_loss = 0
    train_acc = 0

    for batch, (X, y) in enumerate(data_loader):

        X = X.to(device)
        y = y.to(device)

        y_logits = model(X)

        loss = loss_fn(y_logits, y)
        train_loss = train_loss + loss

        acc = accuracy_fn(y, torch.argmax(y_logits, dim=1))
        train_acc = train_acc + acc

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss = train_loss / len(data_loader)
    train_acc = train_acc / len(data_loader)

    return train_loss, train_acc


# Validation loop using val dataloader
def val_step(
    model, 
    dataloader,
    loss_fn,
    device
):

    model.eval()
    val_loss = 0
    val_acc = 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):

            X = X.to(device)
            y = y.to(device)

            y_logits = model(X)

            loss = loss_fn(y_logits, y)
            val_loss = val_loss + loss

            acc = accuracy_fn(y, torch.argmax(y_logits, dim=1))
            val_acc = val_acc + acc

        val_loss = val_loss / len(dataloader)
        val_acc = val_acc / len(dataloader)

    return val_loss, val_acc


def train(
    model, 
    train_dataloader,
    val_dataloader,
    loss_fn,
    optimizer,
    device,
    epochs
):
    
    results = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in torch.arange(epochs):
    
        train_loss, train_acc = train_step(model, 
                                           train_dataloader, 
                                           loss_fn, 
                                           optimizer, 
                                           device)

        val_loss, val_acc = val_step(model, 
                                       val_dataloader, 
                                       loss_fn,
                                       device)
        
        print(
            f'Epoch: {epoch+1} | '
            f'train_loss: {train_loss:.4f} | '
            f'train_acc: {train_acc:.4f} | '
            f'test_loss: {val_loss:.4f} | '
            f'test_acc: {val_acc:.4f}'
        )
