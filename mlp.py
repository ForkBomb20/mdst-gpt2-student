"""MLP Module for GPT-2 Transformer Block"""

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO: Define the fully connected layer for th MLP (hint: it should output 4 times the embedding dimensionality)

        # TODO: Define the projection layer for the MLP (hint: it should output the same dimensionality as the input) (n_embd)

        # Define a GeLU activation function. We use GeLU as it is the activation function used in GPT-2
        self.gelu = nn.GELU()

    def forward(self, x):
        # TODO: Feedforward the input through the fully connected layer, apply the GeLU activation, then project back to the embedding dimensionality
        x = None # Replace this with your code to feedforward through the fully connected layer, you may need more than one line
        return x