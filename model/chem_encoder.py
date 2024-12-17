import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_lib import *

import itertools
import time

class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()

        self.x_size = x_size
        self.h_size = h_size

        # Forget gate
        self.W_f = nn.Linear(x_size + h_size, h_size)
        self.b_f = nn.Parameter(torch.zeros(h_size))

        # Input gate
        self.W_i = nn.Linear(x_size + h_size, h_size)
        self.b_i = nn.Parameter(torch.zeros(h_size))

        # Cell candidate (update)
        self.W_u = nn.Linear(x_size + h_size, h_size)
        self.b_u = nn.Parameter(torch.zeros(h_size))

        # Output gate
        self.W_o = nn.Linear(x_size + h_size, h_size)
        self.b_o = nn.Parameter(torch.zeros(h_size))

    def message_func(self, src, dst):
        """
        Function to generate messages from child nodes.
        Concatenates hidden states (h) and embeddings (embed).
        """
        h = torch.cat([src["h"], src["embed"]], dim=1)
        return {"h": h, "c": src["c"]}

    def reduce_func(self, nodes):
        """
        Function to compute aggregated results from child nodes.
        Calculates forget gate and new cell state.
        """
        # Sum hidden states from children
        h_sum = torch.sum(nodes.mailbox["h"], dim=0)  # Sum over all children
        
        # Forget gate for each child node
        f = torch.sigmoid(self.W_f(h_sum) + self.b_f)  # Shape: (batch_size, h_size)
        
        # Sum up forget gate-weighted cell states
        c_tilde = torch.sum(f * nodes.mailbox["c"], dim=0)  # Weighted sum
        
        return {"h_sum": h_sum, "c": c_tilde}

    def apply_node_func(self, nodes):
        """
        Apply LSTM cell updates for the current node.
        """
        # Input gate
        i = torch.sigmoid(self.W_i(nodes.data["h_sum"]) + self.b_i)
        
        # Cell candidate
        u = torch.tanh(self.W_u(nodes.data["h_sum"]) + self.b_u)
        
        # New cell state
        c = i * u + nodes.data["c"]  # Update cell state
        
        # Output gate
        o = torch.sigmoid(self.W_o(nodes.data["h_sum"]) + self.b_o)
        
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


    def forward(self, node_tensors, adj_matrix_list):
        """
        Forward pass for StructureEncoder.
        """
        device = next(self.parameters()).device

        # Process each tree in the input
        y_list = []
        valid_nodes = []

        for i, (node_tensor, adj_matrix) in enumerate(zip(node_tensors, adj_matrix_list)):
            try:
                # Propagate messages through the tree structure
                res = prop_nodes_topo({
                    'embed': node_tensor[:adj_matrix.size(0), :].to(device),
                    'h': torch.zeros(adj_matrix.size(0), self.h_size).to(device),
                    'c': torch.zeros(adj_matrix.size(0), self.h_size).to(device),
                },
                adj_matrix=adj_matrix,
                processor=self.cell,
                reverse=True,
                update_initial_level=True,
                )

                # Extract root hidden state
                h_root = res['h'][0]  # Root node hidden state
                h_root = h_root.unsqueeze(0)  # Add batch dimension

                # Apply dropout and linear transformations
                h = self.dropout(h_root)
                y = torch.tanh(self.linear(h))
                y2 = self.linear3(torch.relu(self.linear2(h)))
                y = torch.relu(self.linear4(y + y2))
                
                y_list.append(y)
                valid_nodes.append(i)
            except Exception as e:
                pass

        # Concatenate outputs for batch processing
        y = torch.cat(y_list, dim=0)

        return y, valid_nodes
        

