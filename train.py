import torch
import torchvision
import numpy as np
from torch import nn
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_step(model, dataloader, loss_fn, optimizer) :
    model.train()
    train_loss = 0 

    for batch, (x,y) in enumerate(dataloader) :
        x, y = x.to(device), y.to(device)

        #forward pass
        y_pred = model(x) # same as model.forward(x)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        
        #update weight and bias
        loss.backward()
        optimizer.step()

    # adjust to take avg for each batch
    train_loss = train_loss / len(dataloader)
    return train_loss

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    
    model.eval() 
    test_loss = 0
    
    # Turn on inference context manager
    with torch.inference_mode():

        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    return test_loss


def train_model(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.MSELoss(),
          scheduler = None ,
          epochs: int = 5) :
    
    best_loss = np.Infinity
    results = {"train_loss": [],
    }

    for epoch in tqdm(range(epochs)) :
        train_loss = train_step(model, train_dataloader, loss_fn, optimizer)
        scheduler.step(train_loss)

        if  train_loss < best_loss: 
            best_loss = train_loss
            model_name = f'model_mae_{train_loss:.4f}.pth'
            torch.save(model.state_dict(), f'./model_checkpoint/{model_name}')
            print(f' -- save model to ./model_checkpoint/{model_name} with loss {train_loss:.4f}')
    
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
        )

        #Update results dictionary
        results["train_loss"].append(train_loss)
    
    return results