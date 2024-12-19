import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .graph_lib import *

import itertools
import time

class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size, graph_node_dim):
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

        self.edge_one_hot_liner = nn.Linear(3, graph_node_dim)
        self.edge_linear = nn.Linear(graph_node_dim, h_size)

    def message_func(self, src, dst):
        """
        Function to generate messages from child nodes.
        Concatenates hidden states (h) and embeddings (embed).
        """
        joint = src['joint']
        if torch.all(src['joint'] == dst['joint']):
            return {'h': src['h'], 'c': src['c'], 'embed': dst['embed']}
        
        src_node_feature = torch.stack([src['graph'][i][j] for i, j in enumerate(joint[:, 2])])
        dst_node_feature = torch.stack([dst['graph'][i][j] for i, j in enumerate(joint[:, 1])])
        edge_type = torch.stack([F.one_hot((j-1).to(torch.int64), num_classes=3).float() for j in joint[:,3]])
        return {
            "h": src["h"], 
            "c": src["c"], 
            'embed': dst['embed'], 
            'src_node': src_node_feature,
            'dst_node': dst_node_feature,
            'edge_type': edge_type
            }

    def reduce_func(self, nodes):
        """
        Function to compute aggregated results from child nodes.
        Calculates forget gate and new cell state.
        """
        if 'edge_type' in nodes.mailbox:
            edge_w = nodes.mailbox['src_node'] * nodes.mailbox['dst_node']
            edge_w += self.edge_one_hot_liner(nodes.mailbox['edge_type'])
            edge_w = self.edge_linear(edge_w)
            h2 = torch.cat([nodes.mailbox['h'] * edge_w, nodes.mailbox['embed']], dim=1)
        else:
            h2 = torch.cat([nodes.mailbox['h'], nodes.mailbox['embed']], dim=1)

        # Sum hidden states from children
        h_sum = torch.sum(h2, dim=0)  # Sum over all children
        
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
    def __init__(self, x_size, h_size, atom_symbol_size, dropout_rate=0.1):
        super(StructureEncoder, self).__init__()
        self.x_size = x_size
        self.h_size = h_size

        # TreeLSTM cell
        self.cell = ChildSumTreeLSTMCell(x_size, h_size, graph_node_dim=h_size)

        # Vocabulary graph to features
        self.vocab_graph_to_features = VocabGraph(atom_symbol_size, x_size, gat_hidden_dim=x_size//2)

        # Linear transformations
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(h_size, h_size)
        self.linear2 = nn.Linear(h_size, h_size)
        self.linear3 = nn.Linear(h_size, h_size)
        self.linear4 = nn.Linear(h_size, h_size)


    def forward(self, node_tensors, adj_matrix_list, joints_tensor, graph_seq_list):
        """
        Forward pass for StructureEncoder.
        """
        device = next(self.parameters()).device

        # Process each tree in the input
        y_list = []

        for i, (node_tensor, adj_matrix) in enumerate(zip(node_tensors, adj_matrix_list)):
            # try:
            if True:
                features_seq = self.vocab_graph_to_features(graph_seq_list[i][:adj_matrix.size(0)])

                # Propagate messages through the tree structure
                res = prop_nodes_topo({
                    'embed': node_tensor[:adj_matrix.size(0), :].to(device),
                    'joint': joints_tensor[i][:adj_matrix.size(0), :].to(device),
                    'graph': features_seq,
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
            # except Exception as e:
            else:
                pass

        # Concatenate outputs for batch processing
        y = torch.cat(y_list, dim=0)

        return y
            
        
class VocabGraph(nn.Module):
    def __init__(self, atom_symbol_size, feature_size, atom_embedding_dim=4, gat_hidden_dim=16, gat_heads=4, gat_dropout=0.1):
        super(VocabGraph, self).__init__()
        self.atom_embedding = nn.Embedding(atom_symbol_size, atom_embedding_dim)
        self.linear1 = nn.Linear(atom_embedding_dim+3, atom_embedding_dim+3)
        self.gat = GATConv(atom_embedding_dim+3, gat_hidden_dim, heads=gat_heads, dropout=gat_dropout, concat=True)
        self.output_layer = nn.Linear(gat_hidden_dim * gat_heads, feature_size)
    
    def forward(self, graph_seq):
        """
        Convert a vocabulary graph to features.
        """
        device = next(self.parameters()).device
        features_seq = []
        for graph in graph_seq:
            node_tensor, edge_tensor, frag_bond_tensor = graph
            node_tensor = node_tensor.to(device) # Shape: (num_nodes)
            edge_tensor = edge_tensor.to(device) # Shape: (num_edges, 3)
            frag_bond_tensor = frag_bond_tensor.to(device) # Shape: (num_nodes, 3)

            num_nodes = node_tensor.size(0)
            node_embed = self.atom_embedding(node_tensor)
            num_embed = torch.cat([node_embed, frag_bond_tensor], dim=1)
            num_embed = self.linear1(num_embed)

            if num_nodes > 1:
                edge_index = edge_tensor[:, :2].T
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.int64).to(device)
            # adj_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.int32)
            # if num_nodes > 1:
            #     adj_matrix[edge_tensor[:, 0], edge_tensor[:, 1]] = edge_tensor[:, 2]
            #     adj_matrix[edge_tensor[:, 1], edge_tensor[:, 0]] = edge_tensor[:, 2]

            gat_output = self.gat(num_embed, edge_index)
            features = self.output_layer(gat_output)
            features_seq.append(features)
        return features_seq
