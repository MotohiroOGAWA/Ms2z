
import torch
import torch.nn as nn

from .graph_lib import *
from .chem_encoder import *
from .decoder import *

class Ms2z(nn.Module):
    def __init__(
            self, vocab_data, max_seq_len, node_dim, edge_dim,
            latent_dim,
            decoder_layers, decoder_heads, decoder_ff_dim, decoder_dropout,
            fp_dim,
            tgt_token_priority_weight=0.2,
            ):
        super(Ms2z, self).__init__()
        vocab_size = len(vocab_data['token'])
        bos_idx = vocab_data['bos']
        eos_idx = vocab_data['eos']
        pad_idx = vocab_data['pad']
        unk_idx = vocab_data['unk']
        joint_potential = vocab_data['joint_potential']
        tgt_similar = vocab_data['tgt_cosine_matrix']


        self.vocab_size = vocab_size

        self.max_seq_len = max_seq_len
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.latent_dim = latent_dim

        self.tgt_token_priority_weight = tgt_token_priority_weight

        # Vocabulary embedding
        self.vocab_embedding = FragEmbeddings(
            node_dim, edge_dim, 
            self.vocab_size,
            joint_potential,
            )
        self.bos = nn.Parameter(torch.tensor(bos_idx, dtype=torch.int32), requires_grad=False)
        self.eos = nn.Parameter(torch.tensor(eos_idx, dtype=torch.int32), requires_grad=False)
        self.pad = nn.Parameter(torch.tensor(pad_idx, dtype=torch.int32), requires_grad=False)
        self.unk = nn.Parameter(torch.tensor(unk_idx, dtype=torch.int32), requires_grad=False)
        self.tgt_similar = nn.Parameter(tgt_similar, requires_grad=False)

        # Structure encoder
        encoder_h_size = 4*node_dim
        self.chem_encoder = StructureEncoder(node_dim=node_dim, edge_dim=edge_dim, h_size=encoder_h_size)

        # Latent sampler
        self.chem_latent_sampler = LatentSampler(input_dim=encoder_h_size, latent_dim=latent_dim)
        self.linear = nn.Linear(latent_dim, node_dim+edge_dim)

        # Transformer decoder
        self.decoder_layers = decoder_layers
        self.decoder_heads = decoder_heads
        self.decoder_ff_dim = decoder_ff_dim
        self.decoder_dropout = decoder_dropout
        self.decoder = Decoder(decoder_layers, node_dim+edge_dim, edge_dim, decoder_heads, decoder_ff_dim, decoder_dropout)
        self.output_word_layer = nn.Linear(node_dim+edge_dim, self.vocab_size)
        # self.output_joint_layer = nn.Linear(node_dim, self.vocab_embedding.max_joint_cnt)

        # Fingerprint
        self.fp_dim = fp_dim
        self.fp_layer1 = nn.Linear(latent_dim, 2*fp_dim)
        self.fp_layer2 = nn.Linear(2*fp_dim, fp_dim)
        self.fp_dropout = nn.Dropout(0.1)
        self.fp_sigmoid = nn.Sigmoid()

    def forward(self, vocab_tensor, order_tensor, mask_tensor, fp_tensor):
        """
        Forward pass for Ms2z model.

        Args:
            vocab_tensor (torch.Tensor): Vocabulary input tensor.
            order_tensor (list of torch.Tensor): Order tensors for graph adjacency. [batch_size, seq_len, 5 (parent_idx, parent_atom_pos, atom_pos, bond_type[1~3], level)]
            mask_tensor (list of torch.Tensor): Mask tensors.

        Returns:
            z (torch.Tensor): Sampled latent variable.
            mean (torch.Tensor): Mean of the latent variable.
            log_var (torch.Tensor): Log variance of the latent variable.
        """
        # Update vocab_tensor using mask_tensor to insert eos and pad tokens
        token_seq_tensor = torch.cat([
            torch.full((vocab_tensor.size(0), 1), self.bos, dtype=vocab_tensor.dtype, device=vocab_tensor.device), 
            vocab_tensor], dim=1)
        mask_tensor_ex = torch.cat([torch.ones(mask_tensor.size(0),1, dtype=torch.bool, device=mask_tensor.device), mask_tensor], dim=1)
        target_tensor = token_seq_tensor.roll(shifts=-1, dims=1).long()
        for i in range(token_seq_tensor.size(0)):  # Iterate over batch
            eos_index = torch.where(mask_tensor_ex[i] == False)[0]
            if eos_index.numel() > 0:  # If there is at least one False in mask_tensor
                first_eos_index = eos_index[0].item()
                target_tensor[i, first_eos_index-1] = self.eos 
                target_tensor[i, first_eos_index:] = self.pad  
                token_seq_tensor[i, first_eos_index:] = self.pad

        # Process input sequence through the decoder
        padding = torch.empty(mask_tensor.size(0), 1, device=mask_tensor.device, dtype=torch.int64)
        padding.fill_(-1)
        edge_index = torch.where(
            mask_tensor,
            order_tensor[:,:,0],  # Retain original values where mask_tensor is True
            torch.tensor(-2, dtype=torch.int64)  # Replace with self.vocab.pad where mask_tensor is False
        )
        edge_index = torch.cat([padding,edge_index+1], dim=1) # [batch_size, seq_len]
        root_atom_pos_tensor = torch.cat([padding, order_tensor[:,:,2]], dim=1)
        joint_atom_pos_tensor = torch.cat([order_tensor[:,:,1], padding], dim=1)
        root_bond_type_tensor = torch.cat([padding, order_tensor[:,:,3]], dim=1)
        joint_bond_type_tensor = torch.cat([order_tensor[:,:,3], padding], dim=1)
        root_atom_pos_and_type_tensor = torch.stack([root_atom_pos_tensor, root_bond_type_tensor], dim=-1)
        joint_atom_pos_and_type_tensor = torch.stack([joint_atom_pos_tensor, joint_bond_type_tensor], dim=-1)

        node_embed = self.vocab_embedding(token_seq_tensor, root_atom_pos_and_type_tensor)  # Embed vocabulary tensor [batch_size, seq_len+1, node_dim + edge_dim]
        
        token_and_joint_tensor = torch.cat([token_seq_tensor.unsqueeze(-1), joint_atom_pos_and_type_tensor], dim=-1) # [batch_size, seq_len+1, 3]
        edge_attr = self.vocab_embedding.edge_embed(token_and_joint_tensor) # [batch_size, seq_len+1, edge_dim]
        unk_nodes = self.vocab_embedding(self.unk) # [node_dim]


        # Calculate latent variable
        z, mean, log_var = self.calc_chem_z(node_embed[:,1:], edge_attr[:,1:], order_tensor, mask_tensor)

        # Transformer decoder
        memory = z.unsqueeze(1) # [batch_size, 1, latent_dim]
        memory = self.linear(memory)

        # Decode using Transformer Decoder
        memory_mask = torch.zeros(memory.size(0), memory.size(1), dtype=torch.bool).to(memory.device)
        memory_mask[:, 0] = True


        # Decoder
        decoder_output = self.decoder(
            x=node_embed,
            edge_index=edge_index,
            edge_attr=edge_attr,
            enc_output=memory,
            unk_feature=unk_nodes,
            tgt_mask=mask_tensor_ex,
            memory_mask=memory_mask
        ) # Shape: [batch_size, seq_len, node_dim]

        # Flatten for output layer
        word_embeddings_flat = decoder_output.view(-1, decoder_output.size(-1)) # Shape: [batch_size*seq_len, node_dim]

        
        tgt_flat = target_tensor.view(-1)
        tgt_bond_pos_tensor = torch.cat([order_tensor[:,:,2], padding], dim=1)
        tgt_bond_pos_flat = tgt_bond_pos_tensor.view(-1)
        valid_mask = tgt_flat != self.pad

        word_embeddings_flat = word_embeddings_flat[valid_mask]
        tgt_flat = tgt_flat[valid_mask]
        tgt_similar = self.tgt_similar[tgt_flat]
        # tgt_bond_pos_flat = tgt_bond_pos_flat[valid_mask]
        
        logits_vocab = self.output_word_layer(word_embeddings_flat) # Shape: [<= batch_size*seq_len, vocab_size]
        logits_vocab_sigmoid = torch.sigmoid(logits_vocab)
        # logits_bond_pos = self.output_joint_layer(word_embeddings_flat) # Shape: [<= batch_size*seq_len, max_bond_cnt]

        loss_fn1 = nn.CrossEntropyLoss(ignore_index=self.pad)
        loss_fn2 = nn.CrossEntropyLoss(ignore_index=-1)
        similar_token_loss_fn = F.binary_cross_entropy_with_logits
        token_prediction_loss = loss_fn1(logits_vocab, tgt_flat)
        token_similarity_loss = similar_token_loss_fn(logits_vocab_sigmoid, tgt_similar, reduction='none')
        other_weight = (1 - self.tgt_token_priority_weight) / (self.vocab_size - 1)
        token_similarity_loss_weights = torch.full_like(token_similarity_loss, other_weight)
        token_similarity_loss_weights[torch.arange(tgt_flat.size(0)), tgt_flat] = self.tgt_token_priority_weight
        token_similarity_loss = torch.sum(token_similarity_loss * token_similarity_loss_weights, dim=1).mean()
        # bond_pos_prediction_loss = loss_fn2(logits_bond_pos, tgt_bond_pos_flat)

        # KL Divergence loss
        kl_divergence_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        kl_divergence_loss /= mean.size(0)  # Normalize by batch size


        # fingerprint loss
        fp_x = self.fp_layer1(mean)
        fp_x = torch.relu(fp_x)
        fp_x = self.fp_layer2(fp_x)
        fp_x = self.fp_dropout(fp_x)
        fp_x = self.fp_sigmoid(fp_x)

        fp_loss_fn = nn.BCELoss()
        fp_loss = fp_loss_fn(fp_x, fp_tensor)


        # counter loss
        # vocab_idx = torch.where(mask_tensor, vocab_tensor, torch.tensor(-1, dtype=vocab_tensor.dtype))
        # vocab_idx = vocab_idx[vocab_idx != -1].view(-1)
        # atom_counter_loss, inner_bond_counter_loss, outer_bond_cnt_loss = self.vocab_embedding.calc_counter_loss(vocab_idx)

        return token_prediction_loss, token_similarity_loss, kl_divergence_loss, fp_loss

    def calc_chem_z(self, embed_node, edge_attr, order_tensor, mask_tensor):
        """
        Calculate the latent variable for a given input.

        Args:
            vocab_tensor (torch.Tensor): Vocabulary input tensor.
            order_tensor (list of torch.Tensor): Order tensors for graph adjacency. (parent_idx, parent_bond_pos, bond_pos, parent_atom_idx, atom_idx, bond_type_num)
            mask_tensor (list of torch.Tensor): Mask tensors.

        Returns:
            z (torch.Tensor): Sampled latent variable.
        """

        
        # Convert orders and masks into adjacency matrices
        adj_matrix_list = [order_to_adj_matrix(order[:, 0], mask) for order, mask in zip(order_tensor, mask_tensor)]

        # Pass through structure encoder
        encoder_output = self.chem_encoder(embed_node, edge_attr, adj_matrix_list, mask_tensor)

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
            'vocab_data': self.vocab.get_data_to_save(),
            'max_seq_len': self.max_seq_len,
            'node_dim': self.node_dim,
            'edge_dim': self.edge_dim,
            'latent_dim': self.latent_dim,
            'decoder_layers': self.decoder_layers,
            'decoder_heads': self.decoder_heads,
            'decoder_ff_dim': self.decoder_ff_dim,
            'decoder_dropout': self.decoder_dropout,
            'fp_dim': self.fp_dim,
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
            vocab_data=config_param['vocab_data'],
            max_seq_len=config_param['max_seq_len'],
            node_dim=config_param['node_dim'],
            edge_dim=config_param['edge_dim'],
            latent_dim=config_param['latent_dim'],
            decoder_layers=config_param['decoder_layers'],
            decoder_heads=config_param['decoder_heads'],
            decoder_ff_dim=config_param['decoder_ff_dim'],
            decoder_dropout=config_param['decoder_dropout'],
            fp_dim=config_param['fp_dim'],
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



