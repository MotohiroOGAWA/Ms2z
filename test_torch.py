import torch

# Define a tensor with shape [2, 5, 6, 3]
x = torch.tensor([
    [
        [
            [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]
        ],  # seq 0
        [
            [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30], [31, 32, 33], [34, 35, 36]
        ],  # seq 1
        [
            [37, 38, 39], [40, 41, 42], [43, 44, 45], [46, 47, 48], [49, 50, 51], [52, 53, 54]
        ],  # seq 2
        [
            [55, 56, 57], [58, 59, 60], [61, 62, 63], [64, 65, 66], [67, 68, 69], [70, 71, 72]
        ],  # seq 3
    ],
    [
        [
            [91, 92, 93], [94, 95, 96], [97, 98, 99], [100, 101, 102], [103, 104, 105], [106, 107, 108]
        ],  # seq 0
        [
            [109, 110, 111], [112, 113, 114], [115, 116, 117], [118, 119, 120], [121, 122, 123], [124, 125, 126]
        ],  # seq 1
        [
            [127, 128, 129], [130, 131, 132], [133, 134, 135], [136, 137, 138], [139, 140, 141], [142, 143, 144]
        ],  # seq 2
        [
            [145, 146, 147], [148, 149, 150], [151, 152, 153], [154, 155, 156], [157, 158, 159], [160, 161, 162]
        ],  # seq 3
    ],
])


edge_groups = [
    [
        [0, 1, 2],
        [1, 2, 3],
        [3, 4, 5],
        [2, 4, 5],
    ],
    [
        [0, 1, 2],
        [1, 2, 3],
        [3, 4, 5],
        [4, 4, 5],
    ],
]
edge_groups = torch.tensor(edge_groups)

batch_size, seq_len, num_nodes, node_dim = x.size()
num_edges = edge_groups.size(2)

# Define the batch and sequence indices
batch_indices = torch.arange(batch_size)      # Indices for batches
seq_indices = torch.arange(seq_len).unsqueeze(0).tile((batch_size, 1))

# # Use advanced indexing to extract values
# y = torch.stack([x[batch, seq] for batch, seq in zip(batch_indices, seq_indices)])

# # Print the result
# print("y:\n", y)



# # Prepare indices for advanced indexing
# batch_indices_expanded = batch_indices.view(-1, 1).expand(-1, seq_indices.size(1))  # Shape [batch_size, seq_count]

# # Gather the values using advanced indexing
# y = x[batch_indices_expanded, seq_indices]

# # Print the result
# print("y:\n", y)


# Prepare indices for advanced indexing
batch_indices_expanded = batch_indices.view(-1, 1, 1).expand(-1, seq_len, num_edges)  # Shape [batch_size, seq_len, num_edges]
seq_indices_expanded = seq_indices.view(batch_size, -1, 1).expand(-1, -1, num_edges)  # Shape [batch_size, num_nodes, num_edges]
y = x[batch_indices_expanded, seq_indices_expanded, edge_groups] # Shape [batch_size, seq_len, num_edges, node_dim]


print("y:\n", y)