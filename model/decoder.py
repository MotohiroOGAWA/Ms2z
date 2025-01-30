import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import math



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
        """
        Graph Attention Layer (GAT) with batched graphs and optimized node indexing.

        Args:
            in_features (int): Number of input node features.
            out_features (int): Number of output node features.
            edge_dim (int): Dimension of edge attributes.
        """
        super(GATLayer, self).__init__()
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))  # Weight matrix
        self.a = nn.Parameter(torch.Tensor(2 * out_features + edge_dim, 1))  # Attention mechanism
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes the learnable parameters using Xavier initialization.
        """
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, edge_index, edge_attr, node_mask, edge_mask):
        """
        Forward pass for the GAT layer.

        Args:
            x (torch.Tensor): Node feature matrix of shape [batch_size, seq_len, in_features].
            edge_index (torch.Tensor): Graph edge connections of shape [batch_size, num_edges, 2].
            edge_attr (torch.Tensor): Edge attributes of shape [batch_size, num_edges, edge_dim].
            node_mask (torch.Tensor): Boolean mask for invalid nodes [batch_size, seq_len] (True = invalid).
            edge_mask (torch.Tensor): Boolean mask for invalid edges [batch_size, num_edges] (True = invalid).

        Returns:
            torch.Tensor: Updated node embeddings of shape [batch_size, seq_len, out_features].
        """
        batch_size, seq_len, node_dim = x.shape
        _, num_edges, edge_dim = edge_attr.shape

        # **1. Extract valid nodes (where node_mask is False)**
        node_valid_indices = (~node_mask).nonzero(as_tuple=True)  # Shape: [valid_nodes_count, 2]
        node_batch_ids = node_valid_indices[0]  # Batch indices of valid nodes
        valid_node_ids = node_valid_indices[1]  # Node indices within each batch

        # **2. Compute the cumulative offset for valid nodes in each batch**
        valid_counts_per_batch = (~node_mask).sum(dim=1)  # Shape: [batch_size]
        batch_offsets = torch.cumsum(valid_counts_per_batch, dim=0)  # Compute cumulative sum
        batch_offsets = torch.cat([torch.tensor([0], device=x.device), batch_offsets[:-1]])  # Shift for indexing

        # **3. Assign unique global node IDs based on valid nodes only**
        global_node_ids = torch.arange(len(valid_node_ids), device=x.device)

        global_to_batch_node_map = torch.stack([node_batch_ids, valid_node_ids], dim=1)
        batch_node_to_global_map = -torch.ones((batch_size, seq_len), dtype=torch.long, device=x.device)
        batch_node_to_global_map[node_batch_ids, valid_node_ids] = global_node_ids  


        edge_valid_indices = (~edge_mask).nonzero(as_tuple=True)  # Shape: [valid_edges_count, 2]
        edge_batch_ids = edge_valid_indices[0]
        valid_edge_ids = edge_valid_indices[1]

        edge_index_flat = edge_index[edge_batch_ids, valid_edge_ids]  # Shape: [valid_edges_count, 2]
        edge_attr_flat = edge_attr[edge_batch_ids, valid_edge_ids]  # Shape: [valid_edges_count, edge_dim]
        
        src = edge_index_flat[:, 0]
        src = batch_node_to_global_map[edge_batch_ids, src]
        dst = edge_index_flat[:, 1]
        dst = batch_node_to_global_map[edge_batch_ids, dst]

        x_flat = x.reshape(-1, node_dim)[(~node_mask).reshape(-1)] # Shape: [valid_nodes_count, in_features]
        x_transformed = torch.matmul(x_flat, self.W)  # Shape: [batch_size * seq_len, out_features]
        x_transformed = x_transformed[global_node_ids]  # Only use valid nodes

        h_src = x_transformed[src]  # Shape: [valid_edges_count, out_features]
        h_dst = x_transformed[dst]  # Shape: [valid_edges_count, out_features]

        edge_features = torch.cat([h_src, h_dst, edge_attr_flat], dim=-1)  # Shape: [valid_edges_count, 2*out_features + edge_dim]
        
        e = torch.matmul(edge_features, self.a).squeeze(-1)  # Shape: [valid_edges_count]
        e = self.leaky_relu(e)
        
        # Compute the number of edges per batch
        batch_counts = torch.bincount(edge_batch_ids, minlength=batch_size)  # Shape: [batch_size]
        
        # Create tensor for softmax outputs
        attention_coefficients = torch.zeros_like(e)

        # Get start and end indices for each batch
        batch_offsets = torch.cumsum(batch_counts, dim=0)  # Shape: [batch_size]
        batch_offsets = torch.cat([torch.tensor([0], device=x.device), batch_offsets[:-1]])  # Shift to get start indices

        softmax_values = torch.zeros_like(e)
        split_attention_scores = torch.split(e, batch_counts.tolist())  # Split per batch

        softmax_values_split = [F.softmax(batch_scores, dim=0) for batch_scores in split_attention_scores]
        softmax_values = torch.cat(softmax_values_split, dim=0)

        attention_coefficients = torch.zeros_like(e)
        attention_coefficients[edge_batch_ids] = softmax_values

        # **9. Aggregate neighbor information using attention coefficients**
        h_aggregated = torch.zeros(len(global_node_ids), x_transformed.shape[-1], device=x.device)  # Shape: [valid_nodes_count, out_features]
        h_aggregated.index_add_(0, dst, attention_coefficients.unsqueeze(-1) * h_src)

        # **10. Restore output to original shape considering node_mask**
        output = torch.zeros(batch_size * seq_len, x_transformed.shape[-1], device=x.device)  # Shape: [batch_size * seq_len, out_features]
        output[global_node_ids] = h_aggregated  # Assign values to valid nodes

        return output.reshape(batch_size, seq_len, -1)  # Shape: [batch_size, seq_len, out_features]



class HierGATBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, heads, ff_dim, dropout=0.1):
        super(HierGATBlock, self).__init__()
        self.gat = GATLayer(node_dim, node_dim, edge_dim)
        self.self_attention = MultiHeadAttentionBlock(node_dim, h=heads, dropout=dropout)
        self.enc_dec_attention = MultiHeadAttentionBlock(node_dim, h=heads, dropout=dropout)
                
        # Feed Forward Block
        self.feed_forward = FeedForwardBlock(node_dim, ff_dim, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)
        self.norm3 = nn.LayerNorm(node_dim)
        self.norm4 = nn.LayerNorm(node_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, enc_output, tgt_mask, memory_mask, edge_mask):
        # Shape of x [batch_size, seq_len, in_features]
        # Shape of unk_feature [in_features]
        # Shape of enc_output [batch_size, 1, dim]
        x = self.gat(x, edge_index, edge_attr, tgt_mask, edge_mask)
        x = self.norm1(x + self.dropout(x))  # Apply feed-forward block

        x = self.self_attention(x, x, x, query_mask=tgt_mask)
        x = self.norm2(x + self.dropout(x))

        x = self.enc_dec_attention(x, enc_output, enc_output, query_mask=tgt_mask)
        x = self.norm3(x + self.dropout(x))

        x = self.feed_forward(x)
        x = self.norm4(x + self.dropout(x))

        return x  # Shape: [batch_size, seq_len, node_dim]
    

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
    def attention(query, key, value, query_mask=None, dropout=None):
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
        if query_mask is not None:
            # Expand mask to match the attention scores dimensions
            expand_key_mask = query_mask.unsqueeze(1).unsqueeze(2).transpose(2,3).expand(attention_scores.shape)
            attention_scores = attention_scores.masked_fill(expand_key_mask, -1e9)

        # Apply softmax to normalize the scores
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

    def forward(self, q, k, v, query_mask=None):
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
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2).contiguous()
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2).contiguous()
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2).contiguous()

        # Compute attention
        x, self.attention_scores = self.attention(query, key, value, query_mask, self.dropout)

        # Concatenate all heads back together (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, embed_dim)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Final linear transformation (batch_size, seq_len, embed_dim)
        return self.w_o(x)

class HierGATDecoder(nn.Module):
    def __init__(self, num_layers, node_dim, edge_dim, num_heads, ff_dim, dropout=0.1):
        super(HierGATDecoder, self).__init__()
        self.layers = nn.ModuleList([HierGATBlock(node_dim, edge_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x, edge_index, edge_attr, enc_output, tgt_mask, memory_mask, edge_mask):
        """
        Args:
            x (torch.Tensor): Target sequence input (batch_size, tgt_seq_len, embed_dim)
            enc_output (torch.Tensor): Encoder output (batch_size, src_seq_len, embed_dim)
            tgt_mask (torch.Tensor, optional): Mask for the target sequence (Self-Attention mask).
            memory_mask (torch.Tensor, optional): Mask for the encoder output (Encoder-Decoder Attention mask).
        """
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, enc_output, tgt_mask, memory_mask, edge_mask)

        return self.norm(x)
        
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