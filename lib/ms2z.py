
import torch
import torch.nn as nn

from .graph_lib import *
from .chem_encoder import *
from .vocab import Vocab
from .transformer import *

class Ms2z(nn.Module):
    def __init__(
            self, vocab_data, max_seq_len, vocab_embed_dim, 
            latent_dim,
            decoder_layers=2, decoder_heads=8, decoder_ff_dim=512, decoder_dropout=0.1
            ):
        super(Ms2z, self).__init__()
        self.vocab = Vocab.get_vocab_from_data(vocab_data)
        self.vocab_size = len(self.vocab)
        self.max_seq_len = max_seq_len
        self.vocab_embed_dim = vocab_embed_dim
        self.latent_dim = latent_dim

        # Vocabulary embedding
        self.vocab_embedding = nn.Embedding(self.vocab_size, vocab_embed_dim)

        # Structure encoder
        encoder_h_size = vocab_embed_dim
        self.chem_encoder = StructureEncoder(x_size=vocab_embed_dim, h_size=encoder_h_size, atom_symbol_size=len(self.vocab.symbol_to_idx))

        # Latent sampler
        self.chem_latent_sampler = LatentSampler(input_dim=encoder_h_size, latent_dim=latent_dim)
        self.linear = nn.Linear(latent_dim, vocab_embed_dim)

        # Transformer decoder
        self.positional_encoder = PositionalEncoder(vocab_embed_dim, max_seq_len=max_seq_len+1)
        self.decoder = Decoder(decoder_layers, vocab_embed_dim, decoder_heads, decoder_ff_dim, decoder_dropout)
        self.output_word_layer = nn.Linear(vocab_embed_dim, self.vocab_embed_dim)

    def forward(self, vocab_tensor, order_tensor, mask_tensor):
        """
        Forward pass for Ms2z model.

        Args:
            vocab_tensor (torch.Tensor): Vocabulary input tensor.
            order_tensor (list of torch.Tensor): Order tensors for graph adjacency. [seq_len, 3 (parent_idx, parent_bond_pos, bond_pos)]
            mask_tensor (list of torch.Tensor): Mask tensors.

        Returns:
            z (torch.Tensor): Sampled latent variable.
            mean (torch.Tensor): Mean of the latent variable.
            log_var (torch.Tensor): Log variance of the latent variable.
        """
        z, mean, log_var = self.calc_chem_z(vocab_tensor, order_tensor, mask_tensor)

        # Transformer decoder
        memory = z.unsqueeze(1) # [batch_size, 1, latent_dim]
        memory = self.linear(memory)
        memory = self.positional_encoder(memory)

        # Decode using Transformer Decoder
        memory_mask = torch.zeros(memory.size(0), memory.size(1), dtype=torch.bool).to(memory.device)
        memory_mask[:, 0] = True

        # Update vocab_tensor using mask_tensor to insert eos and pad tokens
        input_tensor = torch.cat([
            torch.full((vocab_tensor.size(0), 1), self.vocab.bos, dtype=vocab_tensor.dtype, device=vocab_tensor.device), 
            vocab_tensor], dim=1)
        mask_tensor_ex = torch.cat([torch.ones(mask_tensor.size(0),1, dtype=torch.bool, device=mask_tensor.device), mask_tensor], dim=1)
        target_tensor = input_tensor.roll(shifts=-1, dims=1).long()
        for i in range(input_tensor.size(0)):  # Iterate over batch
            eos_index = torch.where(mask_tensor_ex[i] == False)[0]
            if eos_index.numel() > 0:  # If there is at least one False in mask_tensor
                first_eos_index = eos_index[0].item()
                target_tensor[i, first_eos_index-1] = self.vocab.eos 
                target_tensor[i, first_eos_index:] = self.vocab.pad  
                input_tensor[i, first_eos_index:] = self.vocab.pad  

        
        # Process input sequence through the decoder
        vocab_embeddings = self.vocab_embedding(input_tensor)  # Embed vocabulary tensor
        vocab_embeddings = self.positional_encoder(vocab_embeddings)  # Add positional encoding

        decoder_output = self.decoder(vocab_embeddings, memory, tgt_mask=mask_tensor_ex, memory_mask=memory_mask)
        word_embeddings = self.output_word_layer(decoder_output)  # Shape: [batch_size, seq_len, vocab_embed_dim]
        word_embeddings_flat = word_embeddings.view(-1, self.vocab_embed_dim)

        tgt_word_embeddings = self.vocab_embedding(target_tensor)
        tgt_word_embeddings_flat = tgt_word_embeddings.view(-1, self.vocab_embed_dim)
        valid_mask = target_tensor.view(-1) != self.vocab.pad

        word_embeddings_flat = word_embeddings_flat[valid_mask]
        tgt_word_embeddings_flat = tgt_word_embeddings_flat[valid_mask]

        # Calculate loss components
        token_mismatch_loss = nn.MSELoss()(word_embeddings_flat, tgt_word_embeddings_flat)

        # KL Divergence loss
        kl_divergence_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        kl_divergence_loss /= mean.size(0)  # Normalize by batch size

        return token_mismatch_loss, kl_divergence_loss

    def calc_chem_z(self, vocab_tensor, order_tensor, mask_tensor):
        """
        Calculate the latent variable for a given input.

        Args:
            vocab_tensor (torch.Tensor): Vocabulary input tensor.
            order_tensor (list of torch.Tensor): Order tensors for graph adjacency. (parent_idx, parent_bond_pos, bond_pos, parent_atom_idx, atom_idx, bond_type_num)
            mask_tensor (list of torch.Tensor): Mask tensors.

        Returns:
            z (torch.Tensor): Sampled latent variable.
        """
        # Embed the input vocabulary
        vocab_emb = self.vocab_embedding(vocab_tensor)

        # Convert orders and masks into adjacency matrices
        adj_matrix_list = [order_to_adj_matrix(order[:, 0], mask) for order, mask in zip(order_tensor, mask_tensor)]
        joints_tensor = order_tensor[:, :, [0,3,4,5]] # [batch_size, max_seq_len, 4 (parent_idx, parent_atom_idx, atom_idx, bond_type_num)]
        
        graph_list = [] # [batch_size, num_nodes, 3(node_tensor,edge_tensor,frag_bond_tensor)]
        for i, v_tensor in enumerate(vocab_tensor):
            graph_list.append([self.vocab.vocab_idx_to_graph[int(v_idx)] for j, v_idx in enumerate(v_tensor) if mask_tensor[i][j]])

        # Pass through structure encoder
        encoder_output = self.chem_encoder(vocab_emb, adj_matrix_list, joints_tensor, graph_list)

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



