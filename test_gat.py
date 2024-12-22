import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import math

# from lib.transformer import MultiHeadAttentionBlock

# 0: CCCC
# 1: C(C)C(C)C
# Step 1: Prepare Graph Data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 2
seq_len = 6
node_dim = 8
edge_dim = 4

mask_tensor = torch.tensor(
    [
        [True,True,True,True,False,False],
        [True,True,True,True,True,False],
    ],
    dtype=torch.bool,
    device=device
)

begin_bond_pos_tensor = torch.tensor(
    [
        [-1, 0, 0, 0, -1, -1], 
        [-1, 0, 0, 0, 0, -1], 
    ],
    dtype=torch.long,
    device=device
) # [batch_size, seq_len]

# parent_idx, parent_bond_pos
joint_bond_pos_tensor = torch.tensor(
    [
        [[-1, -1], [0, 1], [1, 1], [2, 1], [-1, -1], [-1, -1]],
        [[-1, -1], [0, 1], [0, 2], [2, 1], [2, 2], [-1, -1]],
    ],
    dtype=torch.long,
    device=device
) # [batch_size, seq_len, 2]

node_features = torch.randn((batch_size, seq_len, node_dim))  # [batch_size, num_nodes, node_dim]

# Initialize output tensors
edge_index = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=device)  # Initialize with -1
bond_pos_tensor = torch.full((batch_size, seq_len, edge_dim), 0.0, dtype=torch.float32, device=device)  # Initialize with -1

for batch in range(batch_size):
    for i in range(seq_len):
        parent_idx, parent_bond_pos = joint_bond_pos_tensor[batch, i]
        if parent_idx >= 0:  # Valid parent connection
            edge_index[batch, i] = parent_idx  # Parent node
            # bond_pos_tensor[batch, i] = parent_bond_pos
        if mask_tensor[batch, i]:  # Valid node
            bond_pos_tensor[batch, i] = torch.randn(edge_dim, dtype=torch.float32, device=device) # Random bond position

node_labels = torch.tensor(
    [
        [1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0],
    ],
    dtype=torch.long, device=device)  # [batch_size, seq_len]

enc_output = torch.randn((batch_size, 1, node_dim))  # [batch_size, 1, node_dim]

data = Data(x=node_features, edge_index=edge_index, edge_attr=bond_pos_tensor, y=node_labels)

# Step 2: Define a GAT Layer
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

# Step 3: Define the GAT Model
class GAT(nn.Module):
    def __init__(self, embed_dim, edge_dim):
        super(GAT, self).__init__()
        self.gat1 = GATLayer(embed_dim, embed_dim, edge_dim)
        self.self_attention = MultiHeadAttentionBlock(embed_dim, h=2, dropout=0.1)
        self.enc_dec_attention = MultiHeadAttentionBlock(embed_dim, h=2, dropout=0.1)
        self.gat2 = GATLayer(embed_dim, embed_dim, edge_dim)
        

    def forward(self, x, edge_index, edge_attr, enc_output, unk_feature, mask=None):
        # Shape of x [batch_size, seq_len, in_features]
        # Shape of unk_feature [in_features]
        # Shape of enc_output [batch_size, 1, dim]
        x, edge_index, edge_attr, node_mask, edge_mask = self.expand_data(x, edge_index, edge_attr, unk_feature, mask)
        enc_output = enc_output.unsqueeze(1) # Expand enc_output to match the dimensions of x
        _valid_edges = None
        for i in range(3):
            x,_valid_edges = self.gat1(x, edge_index, edge_attr, node_mask, edge_mask, _valid_edges)
            # x = F.elu(x)  # Apply non-linear activation
            x = self.self_attention(x, x, x, node_mask)
            x = self.enc_dec_attention(x, enc_output, enc_output, node_mask)
            x,_valid_edges = self.gat2(x, edge_index, edge_attr, node_mask, edge_mask, _valid_edges)
        return x  # Shape: [batch_size, seq_len, seq_len+1, node_dim]
    

    
    def expand_data(self, x:torch.Tensor, edge_index:torch.Tensor, edge_attr: torch.Tensor, unk_feature:torch.Tensor, mask:torch.Tensor=None):
        """
        Expand the input data for use in attention-based models. This includes expanding node features,
        edge indices, edge attributes, and masks to support batch processing and additional dimensions.

        Args:
            x (torch.Tensor): Node features, shape [batch_size, seq_len, in_features].
            edge_index (torch.Tensor): Edge indices, shape [batch_size, seq_len].
            edge_attr (torch.Tensor): Edge attributes, shape [batch_size, seq_len, edge_dim].
            unk_feature (torch.Tensor): Unknown node feature, shape [in_features].
            mask (torch.Tensor, optional): Mask to specify valid nodes, shape [batch_size, seq_len].

        Returns:
            ex_x (torch.Tensor): Expanded node features, shape [batch_size, seq_len, seq_len+1, in_features].
            ex_edge_index (torch.Tensor): Expanded edge indices, shape [batch_size, seq_len, seq_len, 2].
            ex_edge_attr (torch.Tensor): Expanded edge attributes, shape [batch_size, seq_len, seq_len, edge_dim].
            expand_mask_node (torch.Tensor): Expanded mask for nodes, shape [batch_size, seq_len, seq_len+1].
            ex_edge_mask (torch.Tensor): Expanded mask for edges, shape [batch_size, seq_len, seq_len].
        """
        batch_size, seq_len, in_features = x.size()

        # Create a lower-triangular mask for attention (expand_mask)
        expand_mask = self.create_expand_mask(mask, fill=True) # Shape: [batch_size, seq_len, seq_len]
        
        # Expand node_features to match the expanded mask's dimensions
        ex_x = x.unsqueeze(1).expand(-1, seq_len, -1, -1)  # Expand x to [batch_size, seq_len, seq_len, in_features]
        ex_x = torch.where(expand_mask.unsqueeze(-1), ex_x, torch.zeros_like(ex_x))  # Create a zero tensor with the same shape
        # Append an additional dimension for future expansion
        ex_x = torch.cat([ex_x, torch.zeros(batch_size, seq_len, 1, in_features, device=x.device)], dim=2) # Shape: [batch_size, seq_len, seq_len+1, in_features]

        # Expand the unknown feature tensor based on the diagonal mask
        expand_diagonal_mask = self.create_expand_mask(mask, fill=False)  # Expand mask to [batch_size, seq_len, seq_len]
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
        
        ex_edge_mask = self.create_expand_mask(edge_mask, fill=True) # Shape: [batch_size, seq_len, seq_len]
        ex_src = torch.where(ex_edge_mask, src.unsqueeze(1).expand(-1, seq_len, -1), torch.full_like(ex_edge_mask, -1, dtype=edge_index.dtype))  # Shape: [batch_size, seq_len, seq_len]
        ex_tgt = torch.where(ex_edge_mask, tgt.unsqueeze(1).expand(-1, seq_len, -1), torch.full_like(ex_edge_mask, -1, dtype=edge_index.dtype))  # Shape: [batch_size, seq_len, seq_len]

        # If src is -1, tgt is disconnected.
        ex_src = torch.where(((ex_edge_mask & (ex_src == -1))), ex_tgt, ex_src)  # Replace invalid src indices with tgt indices

        ex_edge_index = torch.cat([ex_src.unsqueeze(-1), ex_tgt.unsqueeze(-1)], dim=-1)  # Shape: [batch_size, seq_len, seq_len, 2]

        # Edge attributes
        ex_edge_attr = edge_attr.unsqueeze(1).expand(-1, seq_len, -1, -1)
        ex_edge_attr = torch.where(ex_edge_mask.unsqueeze(-1).expand(-1,-1,-1,edge_attr.size(2)), ex_edge_attr, torch.zeros_like(ex_edge_attr))  # Shape: [batch_size, seq_len, seq_len, edge_dim]

        return ex_x, ex_edge_index, ex_edge_attr, expand_mask_node, ex_edge_mask

                    
    def create_expand_mask(self, mask, fill=True):
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
            expand_mask = mask.unsqueeze(2).unsqueeze(3).transpose(3, 4).expand(attention_scores.shape)
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
        query = query.view(query.shape[0], query.shape[1], query.shape[2], self.h, self.d_k).transpose(2, 3)
        key = key.view(key.shape[0], key.shape[1], key.shape[2], self.h, self.d_k).transpose(2, 3)
        value = value.view(value.shape[0], value.shape[1], value.shape[2], self.h, self.d_k).transpose(2, 3)

        # Compute attention
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)

        # Concatenate all heads back together (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, embed_dim)
        x = x.transpose(2, 3).contiguous().view(x.shape[0], q.shape[1], q.shape[2], self.h * self.d_k)

        # Final linear transformation (batch_size, seq_len, embed_dim)
        return self.w_o(x)

# Step 4: Train and Predict
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    unk_feature = torch.randn(node_dim).to(device)
    out = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device), enc_output.to((device)), unk_feature, mask_tensor)
    loss = F.nll_loss(out, data.y)  # Cross-entropy loss
    loss.backward()
    optimizer.step()
    return loss.item()

def predict(model, data, node_idx):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        return out[node_idx].argmax(dim=0).item()


# Initialize and train the model
model = GAT(embed_dim=node_dim, edge_dim=edge_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(100):
    loss = train(model, data, optimizer)
    print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')

# Predict the next connection for a specific node
target_node = 0
predicted_label = predict(model, data, target_node)
print(f"Predicted label for node {target_node}: {predicted_label}")
