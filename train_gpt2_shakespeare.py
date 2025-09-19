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
    B, T = 4, 32  # we will adjust these later
    train_loader = DataLoaderLite(B, T)

    # Initialize the GPT model
    model = GPT(GPTConfig())
    model.to(device)

    epochs = 50

    # TODO: Define an AdamW optimizer with a learning rate of 3e-4
    optimizer = None 

    # TODO: Create a training loop
    for i in range(epochs):
        optimizer.zero_grad()

        # TODO: Get a batch of data from the DataLoaderLite
        x, y = None
        x, y, = x.to(device), y.to(device)

        # TODO: Forward pass through the model
        logits, loss = None

        # TODO: Propagate the loss back through the model

        
        optimizer.step()
        print(f"Epoch {i + 1}/{epochs}, Loss: {loss.item()}")

if __name__ == "__main__":
    main()