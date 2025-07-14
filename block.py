"""Attention Block for Transformer model."""

import torch
import torch.nn as nn
from attention import CausalSelfAttention
from mlp import MLP

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        # TODO: Define the two layer normalization layers and the attention layer
        self.ln_1 = None
        self.attn = None
        self.ln_2 = None
        
        # Need to define custom MLP because it is not just one feedforward layer but many in paralell for each token.
        # TODO: Define the MLP layer
        self.mlp = None

    def forward(self, x):
        # TODO: Feedforward the result through the attention layer and the MLP layer to get the final output
        x = None # Will need more than just this line
        return x