import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .graph_lib import *

import itertools
import time

class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, node_dim, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()

        self.x_size = node_dim
        self.h_size = h_size

        # Forget gate
        self.W_f = nn.Linear(node_dim + h_size, h_size)
        self.b_f = nn.Parameter(torch.zeros(h_size))

        # Input gate
        self.W_i = nn.Linear(node_dim + h_size, h_size)
        self.b_i = nn.Parameter(torch.zeros(h_size))

        # Cell candidate (update)
        self.W_u = nn.Linear(node_dim + h_size, h_size)
        self.b_u = nn.Parameter(torch.zeros(h_size))

        # Output gate
        self.W_o = nn.Linear(node_dim + h_size, h_size)
        self.b_o = nn.Parameter(torch.zeros(h_size))

        self.edge_linear1 = nn.Linear(2*node_dim+3, 2*node_dim+3)
        self.edge_linear2 = nn.Linear(2*node_dim+3, h_size)

        self.node_linear = nn.Linear(node_dim+h_size, node_dim+h_size)

    def message_func(self, src, dst):
        """
        Function to generate messages from child nodes.
        Concatenates hidden states (h) and embeddings (embed).
        """
        return {
            "h": src["h"], 
            "c": src["c"], 
            'embed': dst['embed'], 
            'src_embed': src['embed'],
            'dst_embed': src['parent_embed'],
            'edge_type': src['edge_type'],
            }

    def reduce_func(self, mailbox, mask):
        """
        Function to compute aggregated results from child nodes.
        Calculates forget gate and new cell state.
        """
        edge_type = mailbox['edge_type']
        src_node_embed = mailbox['src_embed']
        dst_node_embed = mailbox['dst_embed']

        edge_w = torch.cat([src_node_embed, dst_node_embed, edge_type], dim=2)
        edge_w = self.edge_linear1(edge_w)
        edge_w = torch.relu(edge_w)
        edge_w = self.edge_linear2(edge_w)

        h = mailbox['h']
        c = mailbox['c']
        embed = mailbox['embed']
        
        h = h * edge_w
        h2 = torch.cat([h, embed], dim=2)
        masked_h2 = self.node_linear(h2) * mask['h'].unsqueeze(-1)

        # Sum hidden states from children
        h_sum = torch.sum(masked_h2, dim=1)  # Sum over all children
        
        # Forget gate for each child node
        f = torch.sigmoid(self.W_f(h_sum) + self.b_f)  # Shape: (batch_size, h_size)
        
        # Sum up forget gate-weighted cell states
        masked_c = c * mask['c'].unsqueeze(-1)
        c_tilde = torch.sum(f.unsqueeze(1) * masked_c, dim=1)  # Weighted sum
        
        return {"h_sum": h_sum, "c": c_tilde}

    def apply_node_func(self, nodes):
        """
        Apply LSTM cell updates for the current node.
        """
        # Input gate
        i = torch.sigmoid(self.W_i(nodes["h_sum"]) + self.b_i)
        
        # Cell candidate
        u = torch.tanh(self.W_u(nodes["h_sum"]) + self.b_u)
        
        # New cell state
        c = i * u + nodes["c"]  # Update cell state
        
        # Output gate
        o = torch.sigmoid(self.W_o(nodes["h_sum"]) + self.b_o)
        
        # New hidden state
        h = o * torch.tanh(c)
        
        return {"h": h, "c": c}



class StructureEncoder(nn.Module):
    def __init__(self, x_size, h_size, dropout_rate=0.1):
        super(StructureEncoder, self).__init__()
        self.x_size = x_size
        self.h_size = h_size

        # TreeLSTM cell
        self.cell = ChildSumTreeLSTMCell(x_size, h_size)

        # Linear transformations
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(h_size, h_size)
        self.linear2 = nn.Linear(h_size, h_size)
        self.linear3 = nn.Linear(h_size, h_size)
        self.linear4 = nn.Linear(h_size, h_size)

    def forward(self, node_tensor, parent_node_tensor, edge_type_tensor, adj_matrix_list, mask_tensor):
        """
        Forward pass for StructureEncoder.
        """
        # Propagate messages through the tree structure
        try:
            device = node_tensor.device
            batch_size = len(adj_matrix_list)
            res = prop_nodes_topo({
                'embed': node_tensor,
                'parent_embed': parent_node_tensor,
                'edge_type': edge_type_tensor,
                'h': torch.zeros(node_tensor.size(0), node_tensor.size(1), self.h_size).to(node_tensor.device),
                'c': torch.zeros(node_tensor.size(0), node_tensor.size(1), self.h_size).to(node_tensor.device),
            },
            adj_matrix_list=adj_matrix_list,
            processor=self.cell,
            mask_tensor=mask_tensor.to(device),
            device=device,
            reverse=True,
            update_initial_level=True,
            )

            # Extract root hidden state
            h_root = torch.stack([res[i]['h'][0] for i in range(batch_size)])  # Root node hidden state

            # Apply dropout and linear transformations
            h = self.dropout(h_root)
            y = torch.tanh(self.linear(h))
            y2 = self.linear3(torch.relu(self.linear2(h)))
            y = torch.relu(self.linear4(y + y2))
        except Exception as e:
            y = torch.zeros(node_tensor.size(0), self.h_size).to(node_tensor.device)

        return y
            