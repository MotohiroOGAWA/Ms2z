import torch

# Define sample shapes
batch_size = 2
seq_len = 3
E = 6  # Number of edges
num_nodes = 5
node_dim = 4

# Sample tensors
x = torch.randn(batch_size, seq_len, num_nodes, node_dim, device='cuda:0')  # Node features
src_indices = torch.tensor([[[0, -1, -1, -1, -1, -1],
                              [2, 1, -1, -1, -1, -1],
                              [3, 4, -1, -1, -1, -1]],
                             [[1, 0, -1, -1, -1, -1],
                              [3, 2, 4, -1, -1, -1],
                              [0, 1, 2, -1, -1, -1]]], device='cuda:0')  # Source indices
tgt_indices = torch.tensor([[[1, -1, -1, -1, -1, -1],
                              [3, 0, -1, -1, -1, -1],
                              [4, 2, -1, -1, -1, -1]],
                             [[2, 3, -1, -1, -1, -1],
                              [0, 1, 4, -1, -1, -1],
                              [3, 4, 0, -1, -1, -1]]], device='cuda:0')  # Target indices
edge_mask = src_indices != -1  # Shape: [batch_size, seq_len, E]

# Adjust src_indices and tgt_indices to handle -1
valid_src_indices = src_indices.clone()
valid_tgt_indices = tgt_indices.clone()
valid_src_indices[~edge_mask] = 0  # Replace -1 with a valid index (0 as placeholder)
valid_tgt_indices[~edge_mask] = 0  # Replace -1 with a valid index (0 as placeholder)

# Expand indices for gather and scatter
src_indices_expanded = valid_src_indices.unsqueeze(-1).expand(-1, -1, -1, node_dim)  # Shape: [batch_size, seq_len, E, node_dim]

# Gather the features corresponding to valid_src_indices
x_gathered = x.gather(2, src_indices_expanded)  # Shape: [batch_size, seq_len, E, node_dim]

# Initialize the output tensor
y = torch.zeros(batch_size, seq_len, E, num_nodes, node_dim, device=x.device)

# Scatter x_gathered into y based on valid_tgt_indices
y.scatter_(3, valid_tgt_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, node_dim), x_gathered.unsqueeze(3))

# Apply edge_mask to zero out invalid positions in y
y = y * edge_mask.unsqueeze(-1).unsqueeze(-1)

# Print results
print("y shape:", y.shape)
print("y:\n", y)
