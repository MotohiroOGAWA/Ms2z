from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
import pandas as pd
import dill
import os
import yaml
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
import re

from .layers import MzEmbeddings, Encoder, Conv1dFlatten, LinearFlatten

class Ms2z(nn.Module):
    def __init__(self, seq_len:int, compound_features:int,
                 embed_dim:int, num_heads:int, ff_dim:int, dropout:float, num_layers:int, 
                 flatten_linear_hidden_sizes:List[int],
                 z_dim:int):
        super(Ms2z, self).__init__()
        self.seq_len = seq_len
        self.compound_features = compound_features
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.flatten_linear_hidden_sizes = flatten_linear_hidden_sizes
        self.z_dim = z_dim
        
        self.embedding_layer = nn.Linear(compound_features, embed_dim)

        # Create the encoder layer
        self.encoder = Encoder(
            num_layers, 
            embed_dim, 
            num_heads, 
            ff_dim, 
            dropout,
            )

        self.flatten_layer = LinearFlatten(embed_dim, z_dim, hidden_dim=self.flatten_linear_hidden_sizes)

    @staticmethod
    def from_config(config: dict) -> Ms2z:
        return Ms2z(
            seq_len=config['seq_len'],
            compound_features=config['compound_features'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            ff_dim=config['ff_dim'],
            dropout=config['dropout'],
            num_layers=config['num_layers'],
            flatten_linear_hidden_sizes=config['flatten_linear_hidden_sizes'],
            z_dim=config['z_dim'],
            target_data_infoes=config['target_data_infoes']
        )
    
    def get_config(self) -> dict:
        return {
            'mode': self.mode,
            'seq_len': self.seq_len,
            'compound_features': self.compound_features,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout,
            'num_layers': self.num_layers,
            'flatten_linear_hidden_sizes': self.flatten_linear_hidden_sizes,
            'z_dim': self.z_dim,
        }

    def calcurate_z(self, x):
        peak = x[0]
        mask = x[1]

        # [batch_size, seq_len, compound_features] -> [batch_size, seq_len, embed_dim]
        embed = self.embedding_layer(peak) 

        # Encode the natural product values
        x = self.encoder(embed, mask)

        # Flatten the output
        z = self.flatten_layer(x)

        return z

    def forward(self, x, tgt_y):
        # Calculate mean and log variance from x using the encoder network
        mean, log_var = self.calcurate_z_parameters(x)

        # Reparameterize to get latent variable z
        z = self.reparameterize(mean, log_var)

        # Calculate reconstruction loss between z and target (tgt_y)
        # Assuming the decoder reconstructs the input and we compare it with tgt_y
        reconstruction_loss = F.mse_loss(z, tgt_y, reduction='sum')

        # KL Divergence loss
        kl_divergence_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        # Total VAE loss is the sum of reconstruction loss and KL divergence loss
        loss = reconstruction_loss + kl_divergence_loss

        return loss

    def calcurate_z_parameters(self, x):
        # Assume that the encoder network outputs the mean and log variance
        # This should be part of a neural network, typically a few linear layers
        mean = self.encoder_mean(x)  # Network to calculate mean from input x
        log_var = self.encoder_log_var(x)  # Network to calculate log variance from input x

        return mean, log_var

    def reparameterize(self, mean, log_var):
        # Apply the reparameterization trick: z = mean + std * epsilon
        std = torch.exp(0.5 * log_var)  # Calculate the standard deviation from log variance
        epsilon = torch.randn_like(std)  # Sample noise from a standard normal distribution
        z = mean + std * epsilon  # Reparameterization trick

        return z
    

def formula_to_dict(formula):
    """
    Converts a chemical formula in string format into a dictionary with element symbols as keys
    and their respective counts as values.

    Args:
        formula (str): The chemical formula (e.g., 'C6H12O6').

    Returns:
        dict: A dictionary with elements as keys and their counts as values (e.g., {'C': 6, 'H': 12, 'O': 6}).
    """
    # Use regular expressions to find all element and count pairs in the formula
    # The pattern looks for an uppercase letter followed by an optional lowercase letter for the element
    # and then an optional number for the count
    matches = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    
    # Initialize an empty dictionary to store element counts
    element_counts = {}
    
    # Iterate over each element and its count
    for element, count in matches:
        # If no number is provided after the element, set the count to 1
        if count == '':
            count = 1
        else:
            # Convert the count to an integer if it is provided
            count = int(count)
        
        # Add the element and its count to the dictionary
        # If the element already exists, increment its count by the new value
        element_counts[element] = element_counts.get(element, 0) + count
    
    # Return the dictionary with the element counts
    return element_counts

class TargetDataInfo():
    def __init__(self, name, type, size, linear):
        self.name = name
        self.type = type
        self.size = size
        self.linear = linear


class MultiLinear(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Args:
            input_size (int): 入力層のサイズ
            hidden_sizes (list of int): 各隠れ層のサイズを格納するリスト
            output_size (int): 出力層のサイズ
        """
        super(MultiLinear, self).__init__()
        
        # Create a list to hold the layers
        layers = []
        node_sizes = []
        node_sizes.append(input_size)
        node_sizes.extend(hidden_sizes)
        node_sizes.append(output_size)

        for i in range(len(node_sizes)):
            if isinstance(node_sizes[i], float):
                if node_sizes[i] < 0:
                    node_sizes[i] = int(-node_sizes[i+1]*node_sizes[i])
                else:
                    node_sizes[i] = int(node_sizes[i-1]*node_sizes[i])
        
        # Create the linear layers
        for i in range(1, len(node_sizes)):
            if i != len(node_sizes)-1:
                layers.append(nn.Linear(node_sizes[i-1], node_sizes[i]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(node_sizes[i-1], node_sizes[i]))
        
        # Use nn.Sequential to group the layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        # Forward pass through the entire model
        return self.model(x)
    
class ScalarToEmbedding(nn.Module):
    def __init__(self, embed_size: int):
        """
        A linear layer that transforms a 1D tensor of scalars into embedding vectors.

        Args:
        - embed_size (int): The size of the embedding vector to output.
        """
        super(ScalarToEmbedding, self).__init__()
        self.linear = nn.Linear(1, embed_size)  # Input dimension is fixed to 1 for scalars
    
    def forward(self, x):
        """
        Forward pass that transforms the input scalar tensor into embedding vectors.

        Args:
        - x (torch.Tensor): Input tensor with shape (batch_size), where each element is a scalar.

        Returns:
        - torch.Tensor: Output embedding tensor with shape (batch_size, embed_size).
        """
        # Reshape the input tensor from [batch_size] to [batch_size, 1]
        x = x.unsqueeze(1)
        
        # Apply the linear transformation to obtain the embedding vector
        embedded_x = self.linear(x)
        return embedded_x