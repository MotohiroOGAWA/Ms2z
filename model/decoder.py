import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import math


class FragEmbeddings(nn.Module):
    def __init__(self, 
                 node_dim: int, edge_dim: int, joint_edge_dim: int,
                 vocab_size: int,
                 joint_potential,
                 fp_tensor,
                 formula_tensor, 
                 bos, eos, pad, unk,
                 ) -> None:
        super().__init__()
        assert vocab_size == joint_potential.size(0), "vocab_size and joint mismatch"
        assert vocab_size == fp_tensor.size(0), "vocab_size and fp mismatch"
        assert vocab_size == formula_tensor.size(0), "vocab_size and formula mismatch"
        assert edge_dim > 3, "edge_dim must be greater than 3"

        self.embed_dim = node_dim
        self.edge_dim = edge_dim
        self.vocab_size = vocab_size
        self.max_joint_cnt = joint_potential.size(1)
        self.embedding = nn.Embedding(vocab_size, node_dim)


        # Create a 2D tensor mapping (token_idx, joint_pos) to edge_idx
        edge_idx_map = torch.full((vocab_size, self.max_joint_cnt), -1, dtype=torch.long)

        # Assign indices directly using a single operation
        valid_indices = torch.where(joint_potential[:, :, 0] != -1)
        edge_idx_map[valid_indices] = torch.arange(valid_indices[0].size(0), dtype=torch.long)
        self.edge_size = valid_indices[0].size(0)
        self.edge_idx_map = nn.Parameter(edge_idx_map, requires_grad=False)
        self.edge_embedding = nn.Embedding(self.edge_size+1, edge_dim-3, padding_idx=0)

        # Generate one-hot vectors for bond types
        one_hot_bond_type = torch.cat([torch.eye(3), torch.zeros(1,3)], dim=0).to(torch.float32) # [4*3]
        self.bond_type = nn.Parameter(one_hot_bond_type, requires_grad=False) # [-, =, #, nan]

        # joint potential
        self.joint_potential = nn.Parameter(joint_potential, requires_grad=False)

        # fingerprint
        self.fp_linear = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, fp_tensor.size(1))
        )
        self.fp_tensor = nn.Parameter(fp_tensor, requires_grad=False)
        self.fp_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        # formula
        self.formula_linear = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, formula_tensor.size(1))
        )
        self.formula_tensor = nn.Parameter(formula_tensor, requires_grad=False)
        self.formula_loss_fn = F.smooth_l1_loss

        # bos, eos, pad, unk
        self.special_tokens_tensor = nn.Parameter(torch.zeros(vocab_size, dtype=torch.long), requires_grad=False)
        self.special_tokens_tensor[bos] = 1
        self.special_tokens_tensor[eos] = 2
        self.special_tokens_tensor[pad] = 3
        self.special_tokens_tensor[unk] = 4
        self.special_tokens_linear = nn.Linear(node_dim, 5)
        self.special_tokens_loss_fn = F.cross_entropy


        self.criterion = nn.MSELoss()

    def forward(self, idx, joint_info = None):
        # (batch, seq_len) --> (batch, seq_len, node_dim+edge_dim)
        x = self.embedding(idx)
        edge_embed = self.joint_embed(idx, joint_info)
        x = torch.cat([x, edge_embed], dim=-1)
        return x
    
    def edge_embed(self, joint_pos_tensor):
        # (:, 3(token_idx, atom_pos, bond_type(0~4))) -> (:, edge_dim)
        # Handle both 2D and 3D joint_pos_tensor
        # Reshape to (-1, 2) to index edge_idx_map
        flat_joint_pos = joint_pos_tensor[...,:2].view(-1, 2)
        
        # Get edge indices and reshape back to the original shape
        edge_idx = self.edge_idx_map[flat_joint_pos[:, 0], flat_joint_pos[:, 1]] + 1
        edge_idx = edge_idx.view(*joint_pos_tensor.shape[:-1])  # Reshape to match input dims
        
        # Get edge embeddings
        edge_embed = self.edge_embedding(edge_idx)

        # Get bond type embeddings
        bond_type = self.bond_type[joint_pos_tensor[..., 2]]

        # Concatenate edge and bond type embeddings
        edge_embed = torch.cat([edge_embed, bond_type], dim=-1)
        return edge_embed
        
    
    def joint_embed(self, idx_tensor, joint_info=None):
        # joint_info: (batch, seq_len, 2(atom_pos, bond_type(0~4)))
        if joint_info is None:
            edge_embed = torch.zeros_like(
                idx_tensor.unsqueeze(-1).expand(*idx_tensor.shape, self.edge_dim),
                dtype=self.edge_embedding.weight.dtype,  # Match the data type
                device=self.edge_embedding.weight.device  # Match the device
                )
        else:
            token_and_joint_tensor = torch.cat([idx_tensor.unsqueeze(-1), joint_info], dim=-1)
            edge_embed = self.edge_embed(token_and_joint_tensor)
        return edge_embed
    
    def train_all_nodes(self, batch_size, shuffle=True):
        # Get all vocab_ids
        vocab_ids = torch.arange(self.vocab_size, device=self.embedding.weight.device)

        # Shuffle vocab_ids if shuffle=True
        if shuffle:
            vocab_ids = vocab_ids[torch.randperm(self.vocab_size, device=vocab_ids.device)]

        # Process in batches
        for i in range(0, self.vocab_size, batch_size):
            # Get the current batch of vocab_ids
            batch_ids = vocab_ids[i:i+batch_size]
            
            # Train the model using the current batch
            fp_loss, formula_loss, special_tokens_loss = self.train_batch_nodes(batch_ids)

            # Yield the current batch loss and associated vocab_ids
            yield fp_loss, formula_loss, special_tokens_loss
    
    def train_batch_nodes(self, vocab_ids):
        # Get embeddings for the current batch
        embeddings = self.embedding(vocab_ids)  # (batch_size, node_dim)

        return self.calc_node_loss(embeddings, vocab_ids)
        

    def calc_node_loss(self, embeddings, vocab_ids):
        fp_loss = self.calc_fp_loss(embeddings, vocab_ids)
        formula_loss = self.calc_formula_loss(embeddings, vocab_ids)
        special_tokens_loss = self.calc_special_tokens_loss(embeddings, vocab_ids)

        return fp_loss, formula_loss, special_tokens_loss
    
    def calc_fp_loss(self, embeddings, vocab_ids):
        # Predict fingerprints using the linear layer
        predicted_fp = self.fp_linear(embeddings)  # (batch_size, fp_tensor.size(1))

        # Get the true fingerprints for the current batch
        true_fp = self.fp_tensor[vocab_ids]  # (batch_size, fp_tensor.size(1))

        # Compute the loss for the current batch
        element_wise_loss  = self.fp_loss_fn(predicted_fp, true_fp)

        # Create masks for 0 and 1 in fp_tensor
        zero_mask = (true_fp == 0)  # Shape: [batch_size, fp_size]
        one_mask = (true_fp == 1)  # Shape: [batch_size, fp_size]

        # Compute the mean loss for zero_mask and one_mask
        zero_loss_mean = torch.sum(element_wise_loss * zero_mask, dim=1) / (zero_mask.sum(dim=1) + 1e-8)  # Shape: [batch_size]
        one_loss_mean = torch.sum(element_wise_loss * one_mask, dim=1) / (one_mask.sum(dim=1) + 1e-8)    # Shape: [batch_size]

        # Compute the final loss as the average of zero_loss_mean and one_loss_mean
        fp_loss = (zero_loss_mean + one_loss_mean).mean()

        # Yield the current batch loss and associated vocab_ids
        return fp_loss
    
    def calc_formula_loss(self, embeddings, vocab_ids):
        # Predict formula using the linear layer
        predicted_formula = self.formula_linear(embeddings)  # (batch_size, formula_tensor.size(1))

        # Get the true formula for the current batch
        true_formula = self.formula_tensor[vocab_ids]  # (batch_size, formula_tensor.size(1))

        # Compute the loss for the current batch
        formula_loss = self.formula_loss_fn(predicted_formula, true_formula)

        return formula_loss
    
    def calc_special_tokens_loss(self, embeddings, vocab_ids):
        # Predict special tokens using the linear layer
        predicted_special_tokens = self.special_tokens_linear(embeddings)  # (batch_size, 5)
        
        # Get the true special tokens for the current batch
        true_special_tokens = self.special_tokens_tensor[vocab_ids]


        unique_sp_tokens, counts = torch.unique(true_special_tokens, return_counts=True)
        token_weights = (1 / counts.float()) / unique_sp_tokens.size(0)
        weight_map = {token.item(): weight for token, weight in zip(unique_sp_tokens, token_weights)}
        weights = torch.tensor([weight_map[token.item()] for token in true_special_tokens], device=true_special_tokens.device)  # Shape: [data_size]

        # Compute the loss for the current batch
        special_tokens_loss = self.special_tokens_loss_fn(predicted_special_tokens, true_special_tokens, reduction='none')
        special_tokens_loss = (special_tokens_loss * weights).sum()

        return special_tokens_loss


    def calc_counter_loss(self, x):
        embed = self.embedding(x)

        atom_counter_pred = self.atom_counter_linear(embed)
        inner_bond_counter_pred = self.inner_bond_counter_linear(embed)
        outer_bond_counter_pred = self.outer_bond_cnt_linear(embed)

        atom_counter_tgt = self.atom_counter[x]
        inner_bond_counter_tgt = self.inner_bond_counter[x]
        outer_bond_counter_tgt = self.outer_bond_cnt[x]

        atom_counter_loss = self.criterion(atom_counter_pred, atom_counter_tgt)
        inner_bond_counter_loss = self.criterion(inner_bond_counter_pred, inner_bond_counter_tgt)
        outer_bond_cnt_loss = self.criterion(outer_bond_counter_pred, outer_bond_counter_tgt)

        return atom_counter_loss, inner_bond_counter_loss, outer_bond_cnt_loss




class FeedForwardBlock(nn.Module):

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, ff_dim) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ff_dim, embed_dim) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, embed_dim) --> (batch, seq_len, d_ff) --> (batch, seq_len, embed_dim)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, edge_dim):
        super(GATLayer, self).__init__()
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))  # Weight matrix
        self.a = nn.Parameter(torch.Tensor(2 * out_features + edge_dim, 1))  # Attention mechanism
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, edge_index, edge_attr, node_mask, edge_mask, _valid_edges=None):
        if _valid_edges is not None:
            batch_indices = _valid_edges['batch_indices']

            src_valid_indices = _valid_edges['src_valid_indices']
            src_mask = _valid_edges['src_mask']

            tgt_valid_indices = _valid_edges['tgt_valid_indices']
            tgt_mask = _valid_edges['tgt_mask']

            src_indices_expanded = _valid_edges['src_indices_expanded']
            valid_src_indices = _valid_edges['valid_src_indices']
            tgt_indices_expanded = _valid_edges['tgt_indices_expanded']
            valid_tgt_indices = _valid_edges['valid_tgt_indices']
        else:
            _o_valid_edges = {}

        batch_size, seq_len, num_nodes, _ = x.size()
        _, _, _, edge_dim = edge_attr.size()

        # Linear transformation
        h_prime = torch.matmul(x, self.W)  # Shape: [batch_size, seq_len, seq_len+1, out_features]
        node_dim = h_prime.size(-1)
        num_edges = edge_index.size(2)

        # Extract src and tgt indices
        src_indices = edge_index[:, :, :, 0]  # Shape: [batch_size, seq_len, E]
        tgt_indices = edge_index[:, :, :, 1]  # Shape: [batch_size, seq_len, E]


        # Initialize src tensor with zeros
        src = torch.zeros(batch_size, seq_len, seq_len, node_dim, dtype=x.dtype, device=x.device)  # Shape: [batch_size, seq_len, seq_len, node_dim]
        if _valid_edges is None:
            # Prepare batch indices
            batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1, 1).expand_as(src_indices)  # Shape: [batch_size, seq_len, E]
            # Clamp indices to handle invalid ones
            src_valid_indices = src_indices.clamp(min=0)  # Replace invalid indices (-1) with 0
            # Extract features from x based on valid_indices
            valid_features = h_prime[batch_indices, torch.arange(seq_len, device=x.device).view(1, -1, 1).expand_as(src_valid_indices), src_valid_indices]
            # Apply mask to determine where to write features into src
            src_mask = src_indices >= 0  # Only valid indices

            _o_valid_edges['batch_indices'] = batch_indices
            _o_valid_edges['src_valid_indices'] = src_valid_indices
            _o_valid_edges['src_mask'] = src_mask
        else:
            valid_features = h_prime[batch_indices, torch.arange(seq_len, device=x.device).view(1, -1, 1).expand_as(src_valid_indices), src_valid_indices]
        src[src_mask] = valid_features[src_mask]

        # Initialize tgt tensor with zeros
        tgt = torch.zeros(batch_size, seq_len, seq_len, node_dim, dtype=x.dtype, device=x.device)  # Shape: [batch_size, seq_len, E, node_dim]
        if _valid_edges is None:
            # Prepare batch indices
            batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1, 1).expand_as(tgt_indices)  # Shape: [batch_size, seq_len, E]
            # Clamp indices to handle invalid ones
            tgt_valid_indices = tgt_indices.clamp(min=0)  # Replace invalid indices (-1) with 0
            # Extract features from x based on valid_indices
            valid_features = h_prime[batch_indices, torch.arange(seq_len, device=x.device).view(1, -1, 1).expand_as(tgt_valid_indices), tgt_valid_indices]
            # Apply mask to determine where to write features into tgt
            tgt_mask = tgt_indices >= 0  # Only valid indices

            _o_valid_edges['tgt_valid_indices'] = tgt_valid_indices
            _o_valid_edges['tgt_mask'] = tgt_mask
        else:
            valid_features = h_prime[batch_indices, torch.arange(seq_len, device=x.device).view(1, -1, 1).expand_as(tgt_valid_indices), tgt_valid_indices]
        tgt[tgt_mask] = valid_features[tgt_mask]

        # Concatenate source and target features
        cat = torch.cat([src, tgt, edge_attr], dim=-1)  # Shape: [batch_size, seq_len, E, 2 * out_features]

        e = torch.matmul(cat, self.a).squeeze(-1)  # Shape: [batch_size, seq_len, E]
        e = self.leaky_relu(e)
        e = e.masked_fill(~edge_mask, float('-inf'))  # Shape: [batch_size, seq_len, E]
        e[~torch.isfinite(e).any(dim=-1)] = 0  # Set all -inf rows to 0

        # Compute attention coefficients using softmax
        alpha = torch.softmax(e, dim=-1)  # Shape: [batch_size, seq_len, E]

        # Adjust src_indices and tgt_indices to handle -1
        if _valid_edges is None:
            valid_src_indices = src_indices.clone()
            valid_tgt_indices = tgt_indices.clone()
            valid_src_indices[~edge_mask] = 0  # Replace -1 with a valid index (0 as placeholder)
            valid_tgt_indices[~edge_mask] = 0  # Replace -1 with a valid index (0 as placeholder)
            # Expand indices for gather and scatter
            src_indices_expanded = valid_src_indices.unsqueeze(-1).expand(-1, -1, -1, node_dim)  # Shape: [batch_size, seq_len, num_edges, node_dim]
            tgt_indices_expanded = valid_tgt_indices.unsqueeze(-1).expand(-1, -1, -1, node_dim)  # Shape: [batch_size, seq_len, num_edges, node_dim]

            _o_valid_edges['src_indices_expanded'] = src_indices_expanded
            _o_valid_edges['valid_src_indices'] = valid_src_indices
            _o_valid_edges['tgt_indices_expanded'] = tgt_indices_expanded
            _o_valid_edges['valid_tgt_indices'] = valid_tgt_indices


        x_src = h_prime.gather(2, src_indices_expanded)  # Shape: [batch_size, seq_len, num_edges, node_dim]
        x_tgt = h_prime.gather(2, tgt_indices_expanded)  # Shape: [batch_size, seq_len, num_edges, node_dim]


        # Expand alpha to match x_src's shape and perform element-wise multiplication
        alpha_expanded = alpha.unsqueeze(-1).expand(-1, -1, -1, node_dim)  # Shape: [batch_size, seq_len, num_edges, node_dim]
        x_src_weighted = x_src * alpha_expanded  # Element-wise multiplication
        x_tgt_weighted = x_tgt * alpha_expanded  # Element-wise multiplication

        # Initialize output tensor with zeros
        y_src = torch.zeros(batch_size, seq_len, num_edges, num_nodes, node_dim, device=x.device)
        y_tgt = torch.zeros(batch_size, seq_len, num_edges, num_nodes, node_dim, device=x.device)

        # Scatter weighted x_src into y based on tgt_indices
        # Shape: [batch_size, seq_len, num_edges, num_nodes, node_dim]
        y_src.scatter_(3, valid_tgt_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, node_dim), x_src_weighted.unsqueeze(3))
        y_tgt.scatter_(3, valid_src_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, node_dim), x_tgt_weighted.unsqueeze(3))

        # Aggregate features using attention coefficients
        # Shape: [batch_size, seq_len, num_nodes, node_dim]
        h_prime_agg = torch.sum(y_src, dim=2) + torch.sum(y_tgt, dim=2)

        h_prime_agg = F.elu(h_prime_agg)

        if _valid_edges is None:
            return h_prime_agg, _o_valid_edges
        else:
            return h_prime_agg, _valid_edges

    @staticmethod
    def expand_data(x:torch.Tensor, edge_index:torch.Tensor, edge_attr: torch.Tensor, unk_feature:torch.Tensor, mask:torch.Tensor=None):
        """
        Expand the input data for use in attention-based models. This includes expanding node features,
        edge indices, edge attributes, and masks to support batch processing and additional dimensions.

        Args:
            x (torch.Tensor): Node features, shape [batch_size, seq_len, embed_dim].
            edge_index (torch.Tensor): Edge indices, shape [batch_size, seq_len].
            edge_attr (torch.Tensor): Edge attributes, shape [batch_size, seq_len, edge_dim].
            unk_feature (torch.Tensor): Unknown node feature, shape [embed_dim].
            mask (torch.Tensor, optional): Mask to specify valid nodes, shape [batch_size, seq_len].

        Returns:
            ex_x (torch.Tensor): Expanded node features, shape [batch_size, seq_len, seq_len+1(num_nodes), embed_dim].
            ex_edge_index (torch.Tensor): Expanded edge indices, shape [batch_size, seq_len, seq_len(num_edges), 2].
            ex_edge_attr (torch.Tensor): Expanded edge attributes, shape [batch_size, seq_len, seq_len(num_edges), edge_dim].
            expand_mask_node (torch.Tensor): Expanded mask for nodes, shape [batch_size, seq_len, seq_len+1(num_nodes)].
            ex_edge_mask (torch.Tensor): Expanded mask for edges, shape [batch_size, seq_len, seq_len(num_edges)].
        """
        batch_size, seq_len, in_features = x.size()

        # Create a lower-triangular mask for attention (expand_mask)
        expand_mask = GATLayer.create_expand_mask(mask, fill=True) # Shape: [batch_size, seq_len, seq_len]
        
        # Expand node_features to match the expanded mask's dimensions
        ex_x = x.unsqueeze(1).expand(-1, seq_len, -1, -1)  # Expand x to [batch_size, seq_len, seq_len, in_features]
        ex_x = torch.where(expand_mask.unsqueeze(-1), ex_x, torch.zeros_like(ex_x))  # Create a zero tensor with the same shape
        # Append an additional dimension for future expansion
        ex_x = torch.cat([ex_x, torch.zeros(batch_size, seq_len, 1, in_features, device=x.device)], dim=2) # Shape: [batch_size, seq_len, seq_len+1, in_features]

        # Expand the unknown feature tensor based on the diagonal mask
        expand_diagonal_mask = GATLayer.create_expand_mask(mask, fill=False)  # Expand mask to [batch_size, seq_len, seq_len]
        expand_unk = torch.where(expand_diagonal_mask.unsqueeze(-1), unk_feature.view(1, 1, 1, -1), torch.zeros_like(unk_feature).view(1, 1, 1, -1))
        # Append an additional dimension for future expansion
        expand_unk = torch.cat([torch.zeros(batch_size, seq_len, 1, in_features, device=x.device), expand_unk], dim=2) # Shape: [batch_size, seq_len, seq_len+1, in_features]

        # Expand the attention mask to include the additional dimension
        expand_mask_node = torch.cat([expand_mask[:, :, 0].unsqueeze(2), expand_mask], dim=2) # Shape: [batch_size, seq_len, seq_len+1]

        ex_x += expand_unk

        # Extract src and tgt indices
        assert torch.all(edge_index[:, 0] == -1), "No edge with -1 is specified for the root node."

        edge_mask = mask.clone() # Shape: [batch_size, seq_len]
        src = torch.roll(edge_index, shifts=-1, dims=1)  # Shape: [batch_size, seq_len]
        tgt = torch.where(edge_mask, torch.arange(1, seq_len + 1, dtype=edge_index.dtype, device=edge_index.device).repeat(batch_size, 1), -1)  # Shape: [batch_size, seq_len]
        
        ex_edge_mask = GATLayer.create_expand_mask(edge_mask, fill=True) # Shape: [batch_size, seq_len, seq_len]
        ex_src = torch.where(ex_edge_mask, src.unsqueeze(1).expand(-1, seq_len, -1), torch.full_like(ex_edge_mask, -1, dtype=edge_index.dtype))  # Shape: [batch_size, seq_len, seq_len]
        ex_tgt = torch.where(ex_edge_mask, tgt.unsqueeze(1).expand(-1, seq_len, -1), torch.full_like(ex_edge_mask, -1, dtype=edge_index.dtype))  # Shape: [batch_size, seq_len, seq_len]

        # If src is -1, tgt is disconnected.
        ex_src = torch.where(((ex_edge_mask & (ex_src == -1))), ex_tgt, ex_src)  # Replace invalid src indices with tgt indices

        ex_edge_index = torch.cat([ex_src.unsqueeze(-1), ex_tgt.unsqueeze(-1)], dim=-1)  # Shape: [batch_size, seq_len, seq_len, 2]

        # Edge attributes
        ex_edge_attr = edge_attr.unsqueeze(1).expand(-1, seq_len, -1, -1)
        ex_edge_attr = torch.where(ex_edge_mask.unsqueeze(-1).expand(-1,-1,-1,edge_attr.size(2)), ex_edge_attr, torch.zeros_like(ex_edge_attr))  # Shape: [batch_size, seq_len, seq_len, edge_dim]

        return ex_x, ex_edge_index, ex_edge_attr, expand_mask_node, ex_edge_mask
    
    @staticmethod
    def create_expand_mask(mask, fill=True):
        """
        Expand a mask from [batch_size, seq_len] to [batch_size, seq_len, seq_len]
        and set positions where j > i (future positions) to False.

        Args:
            mask (torch.Tensor): Input mask of shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Expanded mask of shape [batch_size, seq_len, seq_len].
        """
        batch_size, seq_len = mask.size()
        
        if fill:
            # Create future mask: [seq_len, seq_len], True where j <= i
            seq_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=mask.device))
        else:
            # Create future mask: [seq_len, seq_len], True where j == i
            seq_mask = torch.eye(seq_len, dtype=torch.bool, device=mask.device)
        
        # Expand input mask to [batch_size, seq_len, seq_len]
        expand_mask = mask.unsqueeze(-1).expand(-1, -1, seq_len)
        
        # Combine with the future mask
        expand_mask = expand_mask & seq_mask
        return expand_mask



class HierGATBlock(nn.Module):
    def __init__(self, embed_dim, edge_dim, heads, ff_dim, dropout=0.1):
        super(HierGATBlock, self).__init__()
        self.gat = GATLayer(embed_dim, embed_dim, edge_dim)
        self.self_attention = MultiHeadAttentionBlock(embed_dim, h=heads, dropout=dropout)
        self.enc_dec_attention = MultiHeadAttentionBlock(embed_dim, h=heads, dropout=dropout)
                
        # Feed Forward Block
        self.feed_forward = FeedForwardBlock(embed_dim, ff_dim, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, enc_output, node_mask, edge_mask, _valid_edges=None):
        # Shape of x [batch_size, seq_len, in_features]
        # Shape of unk_feature [in_features]
        # Shape of enc_output [batch_size, 1, dim]
        x,_valid_edges = self.gat(x, edge_index, edge_attr, node_mask, edge_mask, _valid_edges)
        x = self.norm1(x + self.dropout(x))  # Apply feed-forward block

        x = self.self_attention(x, x, x, key_mask=node_mask)
        x = self.norm2(x + self.dropout(x))

        x = self.enc_dec_attention(x, enc_output, enc_output, query_mask=node_mask)
        x = self.norm3(x + self.dropout(x))

        x = self.feed_forward(x)
        x = self.norm4(x + self.dropout(x))

        return x, _valid_edges  # Shape: [batch_size, seq_len, num_nodes, node_dim]
    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, h: int, dropout: float) -> None:
        """
        Multi-Head Attention Block.

        Args:
            embed_dim (int): The dimension of the input embedding.
            h (int): The number of attention heads.
            dropout (float): Dropout rate for attention scores.
        """
        super().__init__()
        self.embed_dim = embed_dim  # Embedding vector size
        self.h = h  # Number of heads
        
        # Ensure the embedding dimension is divisible by the number of heads
        assert embed_dim % h == 0, "embed_dim must be divisible by h"
        
        self.d_k = embed_dim // h  # Dimension of each head's vector

        # Linear layers to project the input into Q, K, and V
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)  # Linear layer for query
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)  # Linear layer for key
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)  # Linear layer for value

        # Output linear layer
        self.w_o = nn.Linear(embed_dim, embed_dim, bias=False)  # Linear layer for the final output

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, query_mask=None, key_mask=None, dropout=None):
        """
        Calculate the scaled dot-product attention.
        
        Args:
            query (torch.Tensor): The query matrix Q.
            key (torch.Tensor): The key matrix K.
            value (torch.Tensor): The value matrix V.
            query_mask (torch.Tensor, optional): The mask to prevent attention to certain positions.
            dropout (nn.Dropout, optional): Dropout layer for attention scores.

        Returns:
            torch.Tensor: The attention-weighted output.
            torch.Tensor: The attention scores.
        """
        d_k = query.shape[-1]  # Dimension of each head
        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask (if provided)
        if key_mask is not None:
            # Expand mask to match the attention scores dimensions
            expand_key_mask = key_mask.unsqueeze(3).unsqueeze(4).transpose(2,4).expand(attention_scores.shape)
            attention_scores = attention_scores.masked_fill(expand_key_mask == False, -1e9)

        # Apply softmax to normalize the scores
        if query_mask is None:
            attention_scores = torch.softmax(attention_scores, dim=-1)

        # if query_mask is not None:
        #     # Expand mask to match the attention scores dimensions
        #     expand_query_mask = query_mask.unsqueeze(3).unsqueeze(4).transpose(2,3).expand(attention_scores.shape)
        #     attention_scores = attention_scores.masked_fill(expand_query_mask == False, 0.0)

        # Apply dropout (if provided)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Compute the attention-weighted output
        return torch.matmul(attention_scores, value), attention_scores

    def forward(self, q, k, v, query_mask=None, key_mask=None):
        """
        Forward pass of the Multi-Head Attention block.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            mask (torch.Tensor, optional): Mask tensor to apply on attention scores.

        Returns:
            torch.Tensor: Output tensor after multi-head attention.
        """
        # Linear projections for Q, K, V
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Split the embeddings into h heads and reshape (batch_size, seq_len, embed_dim) --> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], query.shape[2], self.h, self.d_k).transpose(2, 3).contiguous()
        key = key.view(key.shape[0], key.shape[1], key.shape[2], self.h, self.d_k).transpose(2, 3).contiguous()
        value = value.view(value.shape[0], value.shape[1], value.shape[2], self.h, self.d_k).transpose(2, 3).contiguous()

        # Compute attention
        x, self.attention_scores = self.attention(query, key, value, query_mask, key_mask, self.dropout)

        # Concatenate all heads back together (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, embed_dim)
        x = x.transpose(2, 3).contiguous().view(x.shape[0], q.shape[1], q.shape[2], self.h * self.d_k)

        # Final linear transformation (batch_size, seq_len, embed_dim)
        return self.w_o(x)

class Decoder(nn.Module):
    def __init__(self, num_layers, embed_dim, edge_dim, num_heads, ff_dim, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([HierGATBlock(embed_dim, edge_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, edge_index, edge_attr, enc_output, unk_feature, tgt_mask, memory_mask):
        """
        Args:
            x (torch.Tensor): Target sequence input (batch_size, tgt_seq_len, embed_dim)
            enc_output (torch.Tensor): Encoder output (batch_size, src_seq_len, embed_dim)
            tgt_mask (torch.Tensor, optional): Mask for the target sequence (Self-Attention mask).
            memory_mask (torch.Tensor, optional): Mask for the encoder output (Encoder-Decoder Attention mask).
        """
        # x: (batch_size, tgt_seq_len, embed_dim)
        # enc_output: (batch_size, src_seq_len, embed_dim)

        x, edge_index, edge_attr, node_mask, edge_mask = GATLayer.expand_data(x, edge_index, edge_attr, unk_feature, tgt_mask)
        enc_output = enc_output.unsqueeze(2).expand(-1, -1, enc_output.size(1), -1) # Expand enc_output to match the dimensions of x
        _valid_edges = None
        for layer in self.layers:
            x, _valid_edges = layer(x, edge_index, edge_attr, enc_output, node_mask, edge_mask, _valid_edges)
        
        # Convert the values of the last unknown node for each batch and sequence to the new connected node
        # Shape: [batch_size, seq_len, num_nodes, node_dim] -> [batch_size, seq_len, node_dim]
        # Create an index tensor for the unknown node positions
        unknown_indices = torch.arange(x.size(1), device=x.device).view(1, -1) + 1  # Shape: [1, seq_len]
        # unknown_indices = unknown_indices.unsqueeze(0).expand(x.size(0), -1)  # Shape: [batch_size, seq_len]
        x = x[torch.arange(x.size(0), device=x.device).unsqueeze(1), torch.arange(x.size(1), device=x.device).unsqueeze(0), unknown_indices]  # Shape: [batch_size, seq_len, node_dim]

        # 最後に正規化を適用
        x = self.norm(x)
        # x: (batch_size, tgt_seq_len, embed_dim)
        return x