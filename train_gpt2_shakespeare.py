"""Train the GPT-2 model on a mini shakespeare dataset."""

import torch
import torch.nn.functional as F
from gpt2 import GPT, GPTConfig
from dataloader import DataLoaderLite
from utils import get_device_and_seed

### BEFORE doing this you need to edit the forward function in GPT2 class to include loss calculation

def main():

    device = get_device_and_seed()
        
    # Create a DataLoaderLite instance
    B, T = 4, 32  # batch size, block size
    train_loader = DataLoaderLite(B, T)

    model = GPT(GPTConfig())
    model.to(device)

    epochs = 50

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(epochs):
        x, y = train_loader.next_batch()
        x, y, = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {i + 1}/{epochs}, Loss: {loss.item()}")

if __name__ == "__main__":
    main()