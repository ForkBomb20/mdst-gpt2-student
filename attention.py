"""Implementation of the attention mechanism."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # TODO: Define the "c_attn" layer, which is a linear layer that produces the key query and value vectors (hint: it should output 3 times the embedding dimensionality) (one for each of query, key, and value)
        self.c_attn = None
        # TODO: Define the "c_proj" layer, which is a linear layer that projects the output back to the embedding dimensionality (hint: it should output the same dimensionality as the input) (n_embd)
        self.c_proj = None
        # TODO: Define the number of heads and the embedding dimensionality from the configuration
        self.n_head = None
        self.n_embd = None

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        
        # TODO: Calculate the query, key, and value vectors using the c_attn layer then split them into sep
        qkv = None # Replace this with your code to calculate qkv
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Need to do some tranposing to match the gpt-2 implementation exactly, don't worry about this
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # TODO: Calculate the attention output using scaled dot-product attention from torch.nn.functional (we want causal attention, so set is_causal=True)
        y = None # Replace this with your code to calculate the attention output
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # TODO: Project the output back to the embedding dimensionality using the c_proj layer
        y = None # Replace this with your code to project the output back to the embedding dimensionality
        
        return y