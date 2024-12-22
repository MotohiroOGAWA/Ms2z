import torch

# Example shapes
batch_size, seq_len, num_nodes, node_dim, E = 2, 3, 4, 5, 6

# Input tensors
x = torch.randn(batch_size, seq_len, num_nodes, node_dim)  # [batch_size, seq_len, num_nodes, node_dim]
indices = torch.randint(0, num_nodes, (batch_size, seq_len, E))  # [batch_size, seq_len, E]

# Create y with zeros
y = torch.zeros(batch_size, seq_len, E, num_nodes, node_dim)  # [batch_size, seq_len, E, num_nodes, node_dim]

# Use advanced indexing to fill y
batch_indices = torch.arange(batch_size).view(-1, 1, 1).expand(batch_size, seq_len, E)
seq_indices = torch.arange(seq_len).view(1, -1, 1).expand(batch_size, seq_len, E)
edge_indices = torch.arange(E).view(1, 1, -1).expand(batch_size, seq_len, E)

# Fill y with values from x based on indices
y[batch_indices, seq_indices, edge_indices, indices] = x[batch_indices, seq_indices, indices]

print("Resulting tensor y:\n", y)
