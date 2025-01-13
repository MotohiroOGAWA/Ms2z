import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import *
from .graph_lib import *

import itertools
import time

class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, node_dim, edge_dim, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
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

        # Attention mechanism
        self.attention = nn.Linear(h_size+node_dim, 1)

        self.edge_linear1 = nn.Linear(h_size+node_dim+edge_dim, h_size+node_dim)
        self.edge_linear2 = nn.Linear(h_size+node_dim, h_size+node_dim)

        self.node_linear1 = nn.Linear(node_dim+h_size, node_dim+h_size)
        self.node_linear2 = nn.Linear(node_dim+h_size, node_dim+h_size)

    def message_func(self, src, dst):
        """
        Function to generate messages from child nodes.
        Concatenates hidden states (h) and embeddings (embed).
        """
        return {
            "h": src["h"], 
            "c": src["c"], 
            'embed': dst['embed'], 
            'edge_attr': src['edge_attr'],
            }

    def reduce_func(self, mailbox, mask):
        """
        Function to compute aggregated results from child nodes.
        Calculates forget gate and new cell state.
        """
        edge_attr = mailbox['edge_attr']
        h = mailbox['h']
        c = mailbox['c']
        embed = mailbox['embed']

        h2 = torch.cat([h, embed, edge_attr], dim=2)
        h2 = self.edge_linear1(h2)
        h2 = torch.relu(h2)
        h2 = self.edge_linear2(h2)

        # Attention weights
        attention_scores = self.attention(h2)  # Shape: (batch_size, num_children, 1)
        attention_scores = attention_scores.masked_fill(mask['h'].unsqueeze(-1) == False, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1)

        masked_h2 = self.node_linear1(h2) * mask['h'].unsqueeze(-1)
        weighted_h2 = attention_weights * masked_h2

        # Sum hidden states from children
        h_sum = torch.sum(weighted_h2, dim=1)  # Sum over all children

        h_sum = self.node_linear2(h_sum)
        
        # Forget gate for each child node
        f = torch.sigmoid(self.W_f(h_sum) + self.b_f)  # Shape: (batch_size, h_size)
        # f = F.leaky_relu(self.W_f(h_sum) + self.b_f)  # Shape: (batch_size, h_size)
        
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
        # i = F.leaky_relu(self.W_i(nodes["h_sum"]) + self.b_i, negative_slope=0.01)
        
        # Cell candidate
        u = torch.tanh(self.W_u(nodes["h_sum"]) + self.b_u)
        # u = F.leaky_relu(self.W_u(nodes["h_sum"]) + self.b_u, negative_slope=0.01)
        
        # New cell state
        c = i * u + nodes["c"]  # Update cell state
        
        # Output gate
        o = torch.sigmoid(self.W_o(nodes["h_sum"]) + self.b_o)
        # o = F.leaky_relu(self.W_o(nodes["h_sum"]) + self.b_o, negative_slope=0.01)
        
        # New hidden state
        h = o * torch.tanh(c)
        # h = o * F.leaky_relu(c, negative_slope=0.01)
        
        return {"h": h, "c": c}


class ChildSumTreeLeakyCell(nn.Module):
    def __init__(self, node_dim, edge_dim, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
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

        # Attention mechanism
        self.attention = nn.Linear(h_size+node_dim, 1)

        self.edge_linear1 = nn.Linear(h_size+node_dim+edge_dim, h_size+node_dim)
        self.edge_linear2 = nn.Linear(h_size+node_dim, h_size+node_dim)

        self.node_linear1 = nn.Linear(node_dim+h_size, node_dim+h_size)
        self.node_linear2 = nn.Linear(node_dim+h_size, node_dim+h_size)

    def message_func(self, src, dst):
        """
        Function to generate messages from child nodes.
        Concatenates hidden states (h) and embeddings (embed).
        """
        return {
            "h": src["h"], 
            "c": src["c"], 
            'embed': dst['embed'], 
            'edge_attr': src['edge_attr'],
            }

    def reduce_func(self, mailbox, mask):
        """
        Function to compute aggregated results from child nodes.
        Calculates forget gate and new cell state.
        """
        edge_attr = mailbox['edge_attr']
        h = mailbox['h']
        c = mailbox['c']
        embed = mailbox['embed']

        h2 = torch.cat([h, embed, edge_attr], dim=2)
        h2 = self.edge_linear1(h2)
        h2 = torch.relu(h2)
        h2 = self.edge_linear2(h2)

        # Attention weights
        attention_scores = self.attention(h2)  # Shape: (batch_size, num_children, 1)
        attention_scores = attention_scores.masked_fill(mask['h'].unsqueeze(-1) == False, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1)

        masked_h2 = self.node_linear1(h2) * mask['h'].unsqueeze(-1)
        weighted_h2 = attention_weights * masked_h2

        # Sum hidden states from children
        h_sum = torch.sum(weighted_h2, dim=1)  # Sum over all children

        h_sum = self.node_linear2(h_sum)
        
        # Forget gate for each child node
        # f = torch.sigmoid(self.W_f(h_sum) + self.b_f)  # Shape: (batch_size, h_size)
        f = F.leaky_relu(self.W_f(h_sum) + self.b_f)  # Shape: (batch_size, h_size)
        
        # Sum up forget gate-weighted cell states
        masked_c = c * mask['c'].unsqueeze(-1)
        c_tilde = torch.sum(f.unsqueeze(1) * masked_c, dim=1)  # Weighted sum
        
        return {"h_sum": h_sum, "c": c_tilde}

    def apply_node_func(self, nodes):
        """
        Apply LSTM cell updates for the current node.
        """
        # Input gate
        # i = torch.sigmoid(self.W_i(nodes["h_sum"]) + self.b_i)
        i = F.leaky_relu(self.W_i(nodes["h_sum"]) + self.b_i, negative_slope=0.01)
        
        # Cell candidate
        # u = torch.tanh(self.W_u(nodes["h_sum"]) + self.b_u)
        u = F.leaky_relu(self.W_u(nodes["h_sum"]) + self.b_u, negative_slope=0.01)
        
        # New cell state
        c = i * u + nodes["c"]  # Update cell state
        
        # Output gate
        # o = torch.sigmoid(self.W_o(nodes["h_sum"]) + self.b_o)
        o = F.leaky_relu(self.W_o(nodes["h_sum"]) + self.b_o, negative_slope=0.01)
        
        # New hidden state
        # h = o * torch.tanh(c)
        h = o * F.leaky_relu(c, negative_slope=0.01)
        
        return {"h": h, "c": c}


class ChildSumTreeSimpleCell(nn.Module):
    def __init__(self, node_dim, edge_dim, h_size):
        super(ChildSumTreeSimpleCell, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.h_size = h_size

        # Input gate
        self.W_i = nn.Linear(node_dim + h_size, h_size)
        self.b_i = nn.Parameter(torch.zeros(h_size))

        # Attention mechanism
        self.attention = nn.Linear(h_size+node_dim, 1)

        self.edge_linear1 = nn.Linear(h_size+node_dim+edge_dim, h_size+node_dim)
        self.edge_linear2 = nn.Linear(h_size+node_dim, h_size+node_dim)

        self.node_linear1 = nn.Linear(node_dim+h_size, node_dim+h_size)
        self.node_linear2 = nn.Linear(node_dim+h_size, node_dim+h_size)

    def message_func(self, src, dst):
        """
        Function to generate messages from child nodes.
        Concatenates hidden states (h) and embeddings (embed).
        """
        return {
            "h": src["h"], 
            'embed': dst['embed'], 
            'edge_attr': src['edge_attr'],
            }

    def reduce_func(self, mailbox, mask):
        """
        Function to compute aggregated results from child nodes.
        Calculates forget gate and new cell state.
        """
        edge_attr = mailbox['edge_attr']
        h = mailbox['h']
        embed = mailbox['embed']

        h2 = torch.cat([h, embed, edge_attr], dim=2)
        h2 = self.edge_linear1(h2)
        h2 = torch.relu(h2)
        h2 = self.edge_linear2(h2)

        # Attention weights
        attention_scores = self.attention(h2)  # Shape: (batch_size, num_children, 1)
        attention_scores = attention_scores.masked_fill(mask['h'].unsqueeze(-1) == False, float('-inf'))
        attention_weights = torch.softmax(attention_scores, dim=1)

        masked_h2 = self.node_linear1(h2) * mask['h'].unsqueeze(-1)
        weighted_h2 = attention_weights * masked_h2

        # Sum hidden states from children
        h_sum = torch.sum(weighted_h2, dim=1)  # Sum over all children

        h_sum = self.node_linear2(h_sum)
        
        return {"h_sum": h_sum}

    def apply_node_func(self, nodes):
        """
        Apply LSTM cell updates for the current node.
        """
        # Input gate
        # i = torch.sigmoid(self.W_i(nodes["h_sum"]) + self.b_i)
        h = self.W_i(nodes["h_sum"]) + self.b_i
        
        return {"h": h}

class StructureEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, h_size, dropout_rate=0.1):
        super(StructureEncoder, self).__init__()
        self.node_dim = node_dim
        self.h_size = h_size

        # TreeLSTM cell
        self.cell = ChildSumTreeLSTMCell(node_dim+edge_dim, edge_dim, h_size)
        # self.cell = ChildSumTreeSimpleCell(node_dim+edge_dim, edge_dim, h_size)

        # Linear transformations
        self.dropout = nn.Dropout(dropout_rate)

        self.linear = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size),
            # nn.ReLU(),
            # nn.Linear(h_size, h_size),
            # nn.ReLU(),
            # nn.Linear(h_size, h_size),
        )

    def forward(self, node_tensor, edge_attr, adj_matrix_list, mask_tensor):
        """
        Forward pass for StructureEncoder.
        """
        # Propagate messages through the tree structure
        device = node_tensor.device
        batch_size = len(adj_matrix_list)
        res = prop_nodes_topo({
            'embed': node_tensor,
            'edge_attr': edge_attr,
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
        y = self.linear(h)
        # y = torch.tanh(self.linear(h))
        # y2 = self.linear3(torch.relu(self.linear2(h)))
        # y = torch.relu(self.linear4(y + y2))

        return y
            