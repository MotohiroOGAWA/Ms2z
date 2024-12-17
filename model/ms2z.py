
import torch
import torch.nn as nn

from .graph_lib import *
from .chem_encoder import *

class Ms2z(nn.Module):
    def __init__(self, vocab_size, vocab_embed_dim, latent_dim):
        super(Ms2z, self).__init__()
        self.vocab_size = vocab_size
        self.vocab_embed_dim = vocab_embed_dim
        self.latent_dim = latent_dim

        # Vocabulary embedding
        self.vocab_embedding = nn.Embedding(vocab_size, vocab_embed_dim)

        # Structure encoder
        encoder_h_size = vocab_embed_dim
        self.chem_encoder = StructureEncoder(x_size=vocab_embed_dim, h_size=encoder_h_size)

        # Latent sampler
        self.chem_latent_sampler = LatentSampler(input_dim=encoder_h_size, latent_dim=latent_dim)

    def forward(self, vocab_tensor, order_tensor, mask_tensor):
        """
        Forward pass for Ms2z model.

        Args:
            vocab_tensor (torch.Tensor): Vocabulary input tensor.
            order_tensor (list of torch.Tensor): Order tensors for graph adjacency.
            mask_tensor (list of torch.Tensor): Mask tensors.

        Returns:
            z (torch.Tensor): Sampled latent variable.
            mean (torch.Tensor): Mean of the latent variable.
            log_var (torch.Tensor): Log variance of the latent variable.
        """
        z, mean, log_var = self.calc_chem_z(vocab_tensor, order_tensor, mask_tensor)
        return z
    
    def calc_chem_z(self, vocab_tensor, order_tensor, mask_tensor):
        """
        Calculate the latent variable for a given input.

        Args:
            vocab_tensor (torch.Tensor): Vocabulary input tensor.
            order_tensor (list of torch.Tensor): Order tensors for graph adjacency.
            mask_tensor (list of torch.Tensor): Mask tensors.

        Returns:
            z (torch.Tensor): Sampled latent variable.
        """
        # Embed the input vocabulary
        vocab_emb = self.vocab_embedding(vocab_tensor)

        # Convert orders and masks into adjacency matrices
        adj_matrix_list = [order_to_adj_matrix(order[:, 0], mask) for order, mask in zip(order_tensor, mask_tensor)]

        # Pass through structure encoder
        encoder_output, valids = self.chem_encoder(vocab_emb, adj_matrix_list)

        # Sample from latent space
        z, mean, log_var = self.chem_latent_sampler(encoder_output)

        return z, mean, log_var


    def get_config_param(self):
        """
        Get model configuration.

        Returns:
            dict: Model configuration.
        """
        return {
            'vocab_size': self.vocab_size,
            'vocab_embed_dim': self.vocab_embed_dim,
            'latent_dim': self.latent_dim,
        }
    
    @staticmethod
    def from_config_param(config_param):
        """
        Create model from configuration parameters.

        Args:
            config_param (dict): Model configuration parameters.

        Returns:
            Ms2z: Model instance.
        """
        return Ms2z(
            vocab_size=config_param['vocab_size'],
            vocab_embed_dim=config_param['vocab_embed_dim'],
            latent_dim=config_param['latent_dim'],
        )

class LatentSampler(nn.Module):
    def __init__(self, input_dim, latent_dim):
        """
        LatentSampler: Encodes input into a latent space using mean and log variance.

        Args:
            input_dim (int): Dimension of the input features.
            latent_dim (int): Dimension of the latent space.
        """
        super(LatentSampler, self).__init__()

        # Linear layers for mean and log variance
        self.fc_mean = nn.Linear(input_dim, latent_dim)
        self.fc_log_var = nn.Linear(input_dim, latent_dim)

    def reparameterize(self, mean, log_var):
        """
        Reparameterization trick to sample z.

        Args:
            mean (torch.Tensor): Mean of the latent variable.
            log_var (torch.Tensor): Log variance of the latent variable.

        Returns:
            torch.Tensor: Sampled latent variable z.
        """
        std = torch.exp(0.5 * log_var)  # Standard deviation
        epsilon = torch.randn_like(std)  # Random noise
        return mean + epsilon * std

    def forward(self, encoder_output):
        """
        Forward pass for LatentSampler.

        Args:
            encoder_output (torch.Tensor): Output from the encoder, shape (batch_size, input_dim).

        Returns:
            z (torch.Tensor): Sampled latent variable, shape (batch_size, latent_dim).
            mean (torch.Tensor): Mean of the latent variable, shape (batch_size, latent_dim).
            log_var (torch.Tensor): Log variance of the latent variable, shape (batch_size, latent_dim).
        """
        mean = self.fc_mean(encoder_output)  # Compute mean
        log_var = self.fc_log_var(encoder_output)  # Compute log variance

        z = self.reparameterize(mean, log_var)  # Sample z using the reparameterization trick

        return z, mean, log_var

