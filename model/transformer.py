import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_lib import LSTMGraphEmbedding
from torch_geometric.data import Batch, Data
import math




class FragEmbeddings(nn.Module):
    def __init__(self, 
                 node_dim: int, edge_dim: int,
                 attached_motif_index_map: torch.Tensor, # (motif_size, attach_size)
                 bonding_cnt_tensor: torch.Tensor, # (max(attached_motif_index_map))
                 atom_layer_list: list,
                 lstm_iterations: int,
                 bos, pad, unk,
                 ) -> None:
        super().__init__()
        attached_motif_size = torch.max(attached_motif_index_map)+1
        assert attached_motif_size == bonding_cnt_tensor.size(0), "attached_motif_size and bonding_cnt_tensor size mismatch"
        assert attached_motif_size == len(atom_layer_list), "attached_motif_size and attached_motif_index_map size mismatch"

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.motif_size = attached_motif_index_map.size(0)
        self.att_size = attached_motif_index_map.size(1)
        self.attached_motif_size = attached_motif_size
        self.attached_motif_index_map = nn.Parameter(attached_motif_index_map, requires_grad=False)
        self.bonding_cnt_tensor = nn.Parameter(bonding_cnt_tensor, requires_grad=False)
        self.max_bonding_cnt = torch.max(bonding_cnt_tensor)
        # self.atom_layer_list = atom_layer_list

        atom_node_dim = atom_layer_list[3]['x'].size(1)
        atom_edge_attr_dim = atom_layer_list[3]['edge_attr'].size(1)
        for i, atom_layer in enumerate(atom_layer_list):
            if i < 3:
                continue
            nodes = atom_layer['x']
            edges = atom_layer['edge_index']
            edges_attr = atom_layer['edge_attr']
            assert nodes.size(1) == atom_node_dim, "atom_node_dim mismatch"
            if edges_attr.size(0) > 0:
                assert edges_attr.size(1) == atom_edge_attr_dim, "atom_edge_attr_dim mismatch"
            else:
                edges = torch.zeros(2, 1, dtype=edges.dtype)
                edges_attr = torch.zeros(1, atom_edge_attr_dim, dtype=edges_attr.dtype)

            self.register_buffer(f'atom_layer_x_{i}', nodes)
            self.register_buffer(f'atom_layer_edge_index_{i}', edges)
            self.register_buffer(f'atom_layer_edge_attr_{i}', edges_attr)
        self.atom_node_dim = atom_node_dim
        self.atom_edge_attr_dim = atom_edge_attr_dim

        self.bos = bos
        self.pad = pad
        self.unk = unk
        self.special_token_embedding = nn.Embedding(3, node_dim)

        self.atom_layer = LSTMGraphEmbedding(
            node_dim=self.atom_node_dim, edge_dim=self.atom_edge_attr_dim,
            hidden_dim=node_dim, iterations=lstm_iterations, directed=False,
        )

        self.edge_linear = nn.Sequential(
            nn.Linear(self.max_bonding_cnt, edge_dim),
        )

        self.calc_attached_motif_idx_to_embeddings = {}
        
    def forward(self, idx):
        if idx.size(-1) == 2:
            # (..., 2) --> (..., node_dim)
            return self.embed_attached_motif(idx)
        elif idx.size(-1) == 3:
            # (..., 3) --> (..., node_dim + edge_dim)
            node_embeddings = self.embed_attached_motif(idx[..., :2])
            edge_attr = self.embed_edge_attr(idx)
            return torch.cat([node_embeddings, edge_attr], dim=-1)
        else:
            raise ValueError(f'Invalid input shape: {idx.size()}')
        
    def reset_calc_embeddings(self):
        self.calc_attached_motif_idx_to_embeddings = {}

    def embed_attached_motif(self, idx):
        # (..., 2) --> (..., node_dim)
        original_shape = idx.shape[:-1]  
        flattened_idx = idx.reshape(-1, 2)  # (N, 2)

        # Mask to separate special tokens (0~2) and normal indices
        special_token_mask = flattened_idx[:, 0] <= 2
        normal_token_mask = ~special_token_mask

        # Handle special tokens
        special_tokens = flattened_idx[special_token_mask, 0]  # Extract first column for special tokens
        special_embeddings = self.special_token_embedding(special_tokens)  # (num_special_tokens, node_dim)

        # Handle normal tokens
        unique_idx, inverse_indices = torch.unique(flattened_idx[normal_token_mask], dim=0, return_inverse=True)

        cached_embeddings = []
        new_indices = []
        for i, idx_tuple in enumerate(unique_idx.tolist()):
            idx_tuple = tuple(idx_tuple)
            if idx_tuple in self.calc_attached_motif_idx_to_embeddings:
                cached_embeddings.append(self.calc_attached_motif_idx_to_embeddings[idx_tuple])
            else:
                new_indices.append(i)

        if new_indices:
            new_unique_idx = unique_idx[new_indices]
            new_embeddings = self.embed_atom_layer(new_unique_idx)  # (new_size, node_dim)

            for i, idx_tuple in zip(new_indices, new_unique_idx.tolist()):
                self.calc_attached_motif_idx_to_embeddings[tuple(idx_tuple)] = new_embeddings[i]

            cached_embeddings.extend(new_embeddings)

        normal_embeddings = torch.stack(cached_embeddings, dim=0)  # (unique_datasize, node_dim)

        # Create full embedding tensor
        full_embeddings = torch.zeros(
            (flattened_idx.size(0), self.node_dim),  # (N, node_dim)
            device=idx.device
        )

        # Assign special and normal embeddings
        full_embeddings[special_token_mask] = special_embeddings
        full_embeddings[normal_token_mask] = normal_embeddings[inverse_indices]

        # Reshape back to original shape (..., node_dim)
        final_embeddings = full_embeddings.view(*original_shape, -1)

        return final_embeddings
    
    def embed_edge_attr(self, bond_pos_tensor):
        """
        Convert bond position tensor to one-hot encoded edge attributes.

        Args:
            bond_pos_tensor (torch.Tensor): Tensor of shape (datasize, 3), where each row contains
                                            (motif_idx, attachment_idx, bond_pos).

        Returns:
            torch.Tensor: One-hot encoded edge attributes of shape (datasize, max_bonding_cnt).
        """
        # Extract motif_idx, attach_idx, and bond_pos
        motif_idx = bond_pos_tensor[..., 0]
        attach_idx = bond_pos_tensor[..., 1]
        bond_pos = bond_pos_tensor[..., 2]

        # Get bonding count for each pair (motif_idx, attach_idx)
        indices = self.attached_motif_index_map[motif_idx, attach_idx]
        bond_cnt = self.bonding_cnt_tensor[indices]  # Shape: (datasize,)

        # Create a full tensor filled with -1.0
        broadcast_shape = (*bond_pos_tensor.shape[:-1], self.max_bonding_cnt)  # (..., max_bonding_cnt)
        one_hot_tensor = torch.full(
            broadcast_shape,
            -1.0,
            dtype=torch.float32,
            device=bond_pos_tensor.device
        )

        # Generate index grid for broadcasting
        bond_idx_range = torch.arange(self.max_bonding_cnt, device=bond_pos_tensor.device).view(
            *([1] * (bond_pos_tensor.ndim - 1)), self.max_bonding_cnt
        )  # Shape: (..., max_bonding_cnt)

        # Create mask for valid positions where bond_cnt is greater than the index
        mask = bond_idx_range < bond_cnt.unsqueeze(-1)  # Shape: (..., max_bonding_cnt)

        # Fill valid positions with 0.0
        one_hot_tensor[mask] = 0.0

        # Convert bond_pos to indexing format and set bond positions to 1.0
        bond_pos_expanded = bond_pos.unsqueeze(-1).expand_as(one_hot_tensor)
        bond_mask = bond_idx_range == bond_pos_expanded
        one_hot_tensor[bond_mask] = 1.0

        edge_attr = self.edge_linear(one_hot_tensor) # (datasize, edge_dim)

        return edge_attr


    def get_atom_layer(self, idx):
        """
        Retrieve a Data object from the registered buffers.

        Args:
            idx (int): Index of the desired graph.

        Returns:
            Data: The corresponding Data object reconstructed from buffers.
        """
        x = getattr(self, f"atom_layer_x_{idx}")
        edge_index = getattr(self, f"atom_layer_edge_index_{idx}")
        edge_attr = getattr(self, f"atom_layer_edge_attr_{idx}")
        # if edge_attr.numel() == 0:  # Handle empty edge_attr case
        #     edge_attr = None
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def batched_atom_layer(self, idx_tensor):
        selected_graphs = [self.get_atom_layer(self.attached_motif_index_map[idx[0],idx[1]]) for idx in idx_tensor]
        batched_data = Batch.from_data_list(selected_graphs)
        return batched_data
    
    def embed_atom_layer(self, idx_tensor):
        batched_atom_layer = self.batched_atom_layer(idx_tensor)
        x = self.atom_layer(batched_atom_layer)
        return x
    


    
    # def train_all_nodes(self, batch_size, shuffle=True):
    #     # Get all vocab_ids
    #     vocab_ids = torch.arange(self.vocab_size, device=self.embedding.weight.device)

    #     # Shuffle vocab_ids if shuffle=True
    #     if shuffle:
    #         vocab_ids = vocab_ids[torch.randperm(self.vocab_size, device=vocab_ids.device)]

    #     # Process in batches
    #     for i in range(0, self.vocab_size, batch_size):
    #         # Get the current batch of vocab_ids
    #         batch_ids = vocab_ids[i:i+batch_size]
            
    #         # Train the model using the current batch
    #         fp_loss, formula_loss, special_tokens_loss = self.train_batch_nodes(batch_ids)

    #         # Yield the current batch loss and associated vocab_ids
    #         yield fp_loss, formula_loss, special_tokens_loss
    
    # def train_batch_nodes(self, vocab_ids):
    #     # Get embeddings for the current batch
    #     embeddings = self.embedding(vocab_ids)  # (batch_size, node_dim)

    #     return self.calc_node_loss(embeddings, vocab_ids)
        

    # def calc_node_loss(self, embeddings, vocab_ids):
    #     fp_loss = self.calc_fp_loss(embeddings, vocab_ids)
    #     formula_loss = self.calc_formula_loss(embeddings, vocab_ids)
    #     special_tokens_loss = self.calc_special_tokens_loss(embeddings, vocab_ids)

    #     return fp_loss, formula_loss, special_tokens_loss
    
    # def calc_fp_loss(self, embeddings, vocab_ids):
    #     # Predict fingerprints using the linear layer
    #     predicted_fp = self.fp_linear(embeddings)  # (batch_size, fp_tensor.size(1))

    #     # Get the true fingerprints for the current batch
    #     true_fp = self.fp_tensor[vocab_ids]  # (batch_size, fp_tensor.size(1))

    #     # Compute the loss for the current batch
    #     element_wise_loss  = self.fp_loss_fn(predicted_fp, true_fp)

    #     # Create masks for 0 and 1 in fp_tensor
    #     zero_mask = (true_fp == 0)  # Shape: [batch_size, fp_size]
    #     one_mask = (true_fp == 1)  # Shape: [batch_size, fp_size]

    #     # Compute the mean loss for zero_mask and one_mask
    #     zero_loss_mean = torch.sum(element_wise_loss * zero_mask, dim=1) / (zero_mask.sum(dim=1) + 1e-8)  # Shape: [batch_size]
    #     one_loss_mean = torch.sum(element_wise_loss * one_mask, dim=1) / (one_mask.sum(dim=1) + 1e-8)    # Shape: [batch_size]

    #     # Compute the final loss as the average of zero_loss_mean and one_loss_mean
    #     fp_loss = (zero_loss_mean + one_loss_mean).mean()

    #     # Yield the current batch loss and associated vocab_ids
    #     return fp_loss
    
    # def calc_formula_loss(self, embeddings, vocab_ids):
    #     # Predict formula using the linear layer
    #     predicted_formula = self.formula_linear(embeddings)  # (batch_size, formula_tensor.size(1))

    #     # Get the true formula for the current batch
    #     true_formula = self.formula_tensor[vocab_ids]  # (batch_size, formula_tensor.size(1))

    #     # Compute the loss for the current batch
    #     formula_loss = self.formula_loss_fn(predicted_formula, true_formula)

    #     return formula_loss
    
    # def calc_special_tokens_loss(self, embeddings, vocab_ids):
    #     # Predict special tokens using the linear layer
    #     predicted_special_tokens = self.special_tokens_linear(embeddings)  # (batch_size, 5)
        
    #     # Get the true special tokens for the current batch
    #     true_special_tokens = self.special_tokens_tensor[vocab_ids]


    #     unique_sp_tokens, counts = torch.unique(true_special_tokens, return_counts=True)
    #     token_weights = (1 / counts.float()) / unique_sp_tokens.size(0)
    #     weight_map = {token.item(): weight for token, weight in zip(unique_sp_tokens, token_weights)}
    #     weights = torch.tensor([weight_map[token.item()] for token in true_special_tokens], device=true_special_tokens.device)  # Shape: [data_size]

    #     # Compute the loss for the current batch
    #     special_tokens_loss = self.special_tokens_loss_fn(predicted_special_tokens, true_special_tokens, reduction='none')
    #     special_tokens_loss = (special_tokens_loss * weights).sum()

    #     return special_tokens_loss



class MzEmbeddings(nn.Module):
    def __init__(self, embed_dim: int, total_size: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.total_size = total_size
        self.embedding = nn.Embedding(total_size, embed_dim)

    def forward(self, mz_idx, intensity):
        # (batch, seq_len) --> (batch, seq_len, embed_dim)
        embedding = self.embedding(mz_idx)
        norm = embedding.norm(p=2, dim=1, keepdim=True)
        embedding = embedding / norm
        embedding = embedding * intensity.unsqueeze(2)
        return embedding
 
    
class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, embed_dim)
        return self.dropout(self.embedding(x))

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 6000, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # Make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # Add constant positional encoding to the embedding
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len].detach()  # detach() で計算グラフから切り離す

        if x.is_cuda:  # GPU に転送する場合
            pe = pe.cuda()
            
        x = x + pe  # 位置エンコーディングを加算
        x = self.dropout(x)  # Dropout を適用
        return x

class FeedForwardBlock(nn.Module):

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, ff_dim) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ff_dim, embed_dim) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, embed_dim) --> (batch, seq_len, d_ff) --> (batch, seq_len, embed_dim)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

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
    def attention(query, key, value, mask=None, dropout=None):
        """
        Calculate the scaled dot-product attention.
        
        Args:
            query (torch.Tensor): The query matrix Q.
            key (torch.Tensor): The key matrix K.
            value (torch.Tensor): The value matrix V.
            mask (torch.Tensor, optional): The mask to prevent attention to certain positions.
            dropout (nn.Dropout, optional): Dropout layer for attention scores.

        Returns:
            torch.Tensor: The attention-weighted output.
            torch.Tensor: The attention scores.
        """
        d_k = query.shape[-1]  # Dimension of each head
        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask (if provided)
        if mask is not None:
            # Expand mask to match the attention scores dimensions
            expand_mask = mask.unsqueeze(1).unsqueeze(2).transpose(2, 3).expand(attention_scores.shape)
            attention_scores = attention_scores.masked_fill(expand_mask == True, -1e9)

        # Apply softmax to normalize the scores
        attention_scores = torch.softmax(attention_scores, dim=-1)

        # Apply dropout (if provided)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Compute the attention-weighted output
        return torch.matmul(attention_scores, value), attention_scores

    def forward(self, q, k, v, mask=None):
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
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Compute attention
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)

        # Concatenate all heads back together (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, embed_dim)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Final linear transformation (batch_size, seq_len, embed_dim)
        return self.w_o(x)


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        # Self-Attention Block (causal mask is usually applied here in practice)
        self.self_attention = MultiHeadAttentionBlock(embed_dim, num_heads, dropout)
        
        # Encoder-Decoder Attention Block
        self.enc_dec_attention = MultiHeadAttentionBlock(embed_dim, num_heads, dropout)
        
        # Feed Forward Block
        self.feed_forward = FeedForwardBlock(embed_dim, ff_dim, dropout)  # FeedForwardBlockを使用
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        """
        Args:
            x (torch.Tensor): Target sequence input (batch_size, tgt_seq_len, embed_dim)
            enc_output (torch.Tensor): Encoder output (batch_size, src_seq_len, embed_dim)
            tgt_mask (torch.Tensor, optional): Mask for the target sequence (Self-Attention mask).
            memory_mask (torch.Tensor, optional): Mask for the encoder output (Encoder-Decoder Attention mask).
        """
        # Self-Attention Block
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Encoder-Decoder Attention Block
        enc_dec_attn_output = self.enc_dec_attention(x, enc_output, enc_output, memory_mask)
        x = self.norm2(x + self.dropout(enc_dec_attn_output))
        
        # Feed Forward Block
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        """
        Args:
            x (torch.Tensor): Target sequence input (batch_size, tgt_seq_len, embed_dim)
            enc_output (torch.Tensor): Encoder output (batch_size, src_seq_len, embed_dim)
            tgt_mask (torch.Tensor, optional): Mask for the target sequence (Self-Attention mask).
            memory_mask (torch.Tensor, optional): Mask for the encoder output (Encoder-Decoder Attention mask).
        """
        # x: (batch_size, tgt_seq_len, embed_dim)
        # enc_output: (batch_size, src_seq_len, embed_dim)
        
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)
        
        # 最後に正規化を適用
        x = self.norm(x)
        # x: (batch_size, tgt_seq_len, embed_dim)
        return x




class Conv1dFlatten(nn.Module):
    def __init__(self, embed_dim, out_features, seq_len, conv_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(Conv1dFlatten, self).__init__()
        
        layers = []
        in_channels = embed_dim
        for out_channels in conv_channels:
            layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)), 
            layers.append(nn.ReLU())
            in_channels = out_channels
        
        # 最終的な畳み込み層で必要な出力特徴量に合わせる
        layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_features, kernel_size=seq_len))

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, seq_len, embed_dim) -> (batch, embed_dim, seq_len)
        x = self.conv_layers(x) # (batch, out_features, 1)
        x = x.squeeze(2) # (batch_size, out_features, 1) -> (batch_size, out_features)
        return x
    


class LinearFlatten(nn.Module):
    def __init__(self, embed_dim, out_features, hidden_dim:list[int]=[]):
        super(LinearFlatten, self).__init__()
        
        layers = []
        in_channels = embed_dim
        nodes = [embed_dim] + hidden_dim + [out_features]
        for i in range(len(nodes) - 2):
            layers.append(nn.Linear(in_features=nodes[i], out_features=nodes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=nodes[-2], out_features=nodes[-1]))

        self.linear_layers = nn.Sequential(*layers)

    def forward(self, x):
        # Apply average pooling over the sequence length dimension
        x = x.mean(dim=1)  # Shape: [batch_size, embed_dim]

        # Linear transformation: [batch_size, embed_dim] -> [batch_size, out_features]
        x = self.linear_layers(x)
        return x
    
if __name__ == "__main__":
    # Parameters
    batch_size = 2
    seq_len = 5
    embed_dim = 4
    num_heads = 2

    # Sample input: [batch_size, seq_len, embed_dim]
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Key padding mask: [batch_size, seq_len]
    # True = padding position (ignore), False = valid position (attend to)
    key_padding_mask = torch.tensor([[False, False, True, True, True],  # Only first 2 tokens are valid
                                    [False, False, False, True, True]]) # Only first 3 tokens are valid

    # Define multi-head attention
    multi_head_attn = MultiHeadAttentionBlock(embed_dim=embed_dim, h=num_heads, dropout=0.1)

    # Apply multi-head attention with key_padding_mask
    attn_output, attn_weights = multi_head_attn(x, x, x, mask=key_padding_mask)

    # Print results
    print("Attention Output:")
    print(attn_output)

    print("\nAttention Weights:")
    print(attn_weights)