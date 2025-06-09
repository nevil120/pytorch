import torch

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc


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

    #Run through all the batches
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


def test_step(
    model, 
    data_loader,
    loss_fn,
    device
):

    model.eval()
    test_loss = 0
    test_acc = 0

    with torch.inference_mode():
        #Run through all the batches
        for batch, (X, y) in enumerate(data_loader):

            X = X.to(device)
            y = y.to(device)

            y_logits = model(X)

            loss = loss_fn(y_logits, y)
            test_loss = test_loss + loss

            acc = accuracy_fn(y, torch.argmax(y_logits, dim=1))
            test_acc = test_acc + acc

        test_loss = test_loss / len(data_loader)
        test_acc = test_acc / len(data_loader)

    return test_loss, test_acc


def train(
    model, 
    train_data_loader,
    test_data_loader,
    loss_fn,
    optimizer,
    device,
    epochs
):
    
    results = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for epoch in torch.arange(epochs):
    
        train_loss, train_acc = train_step(model, 
                                           train_data_loader, 
                                           loss_fn, 
                                           optimizer, 
                                           device)

        test_loss, test_acc = test_step(model, 
                                       test_data_loader, 
                                       loss_fn,
                                       device)
        
        print(
            f'Epoch: {epoch+1} | '
            f'train_loss: {train_loss} | '
            f'train_acc: {train_acc} | '
            f'test_loss: {test_loss} | '
            f'test_acc: {test_acc}'
        )
        
        results['train_loss']. append(train_loss)
        results['train_acc']. append(train_acc)
        results['test_loss']. append(test_loss)
        results['test_acc']. append(test_acc)
        
    return results
