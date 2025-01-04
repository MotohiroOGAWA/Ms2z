import torch
import torch.nn as nn
from collections import deque, defaultdict



class Node:
    def __init__(self, node_id, features):
        """
        Initialize a Node object.

        Args:
            node_id (int): Unique identifier for the node.
            features (dict): Dictionary of features for the node. Example: {'x': torch.Tensor, 'y': torch.Tensor}.
        """
        self.id = node_id
        self.mailbox = defaultdict(lambda: torch.empty(0, 0))  # Mailbox to store received messages
        self.data = defaultdict(lambda: torch.empty(0, 0))  # Data dictionary to store node features
        for key, value in features.items():
            self.data[key] = value

    def reset_mailbox(self):
        """Reset the mailbox for the next iteration."""
        self.mailbox = defaultdict(lambda: torch.zeros(0))


def prop_nodes_topo(node_data:dict[torch.Tensor], adj_matrix_list, processor, mask_tensor, device, reverse=False, update_initial_level = False):
    """
    Propagate messages through nodes in topological order.

    Args:
        node_data (dict[torch.Tensor]): A dictionary of node features.
        adj_matrix (torch.Tensor): Adjacency matrix of shape [node_size, node_size].
        processor (object): An object with `message_func`, `reduce_func`, and `apply_node_func` methods.
        reverse (bool): Whether to process nodes in reverse topological order. Default is False.
        update_initial_level (bool): Whether to additionally process the initial parent nodes (levels[0]).

    Returns:
        dict: Updated node features after propagation.
    """
    # Check if node features and adjacency matrix dimensions match
    # node_size = None
    batch_size = len(adj_matrix_list)
    # for name, features in node_data.items():
    #     if node_size is None:
    #         node_size = features.size(0)
    #     elif isinstance(features, torch.Tensor) and features.size(0) != node_size:
    #         raise ValueError("Node size mismatch.")
    #     elif isinstance(features, list) and len(features) != node_size:
    #         raise ValueError("Node size mismatch.")
        
    # if node_size != adj_matrix.size(0) or adj_matrix.size(0) != adj_matrix.size(1):
    #     raise ValueError("Node size and adjacency matrix dimensions do not match.")

    # Topological sort grouped by levels
    # print(1)
    levels_list = [topo_sort_levels(adj_matrix, reverse=reverse) for adj_matrix in adj_matrix_list]

    # Duplicate levels[0] if update_initial_level is True
    for levels in levels_list:
        if update_initial_level and len(levels) > 0:
            levels.insert(0, levels[0])  # Add the initial level again for reprocessing
    
    parent_to_child_id_list = [defaultdict(list) for _ in range(batch_size)]

    # Find all edges (parent -> child) using torch.nonzero
    edges_list = [torch.nonzero(adj_matrix, as_tuple=False) for adj_matrix in adj_matrix_list]

    # Add relations to the nodes
    for i, edges in enumerate(edges_list):
        for edge in edges:
            parent_id, child_id = edge.tolist()
            if reverse:
                if child_id not in parent_to_child_id_list[i]:
                    parent_to_child_id_list[i][child_id] = []
                parent_to_child_id_list[i][child_id].append(parent_id)
            else:
                if parent_id not in parent_to_child_id_list[i]:
                    parent_to_child_id_list[i][parent_id] = []
                parent_to_child_id_list[i][parent_id].append(child_id)

    nodes_list = [[Node(node_id, {key:value[i][node_id] for key, value in node_data.items()}) for node_id in torch.nonzero(mask_tensor[i], as_tuple=False)] for i in range(batch_size)]

    # Process nodes level by level
    result = [defaultdict(dict) for _ in range(batch_size)]
    finish_flag = torch.zeros(batch_size, dtype=torch.bool)
    n = -1

    while not torch.all(finish_flag):
        n += 1
        pair_ids_list = torch.zeros(0, 3, dtype=torch.int64)
        for li, levels in enumerate(levels_list):
            if len(levels) <= n:
                finish_flag[li] = True
                continue
            parent_ids = levels[n]
            if update_initial_level and n == 0:
                pair_ids = torch.tensor([[li, parent_id, parent_id] for parent_id in parent_ids], dtype=torch.int64)
            else:
                pair_ids = torch.tensor([[li, parent_id, child_id] for parent_id in parent_ids for child_id in parent_to_child_id_list[li][parent_id]], dtype=torch.int64)
            pair_ids_list = torch.cat((pair_ids_list, pair_ids), dim=0)

        if torch.all(finish_flag):
            break

        if pair_ids_list.size(0) == 0:
            continue

        src = {name: torch.stack([features[li,i] for li, i in pair_ids_list[:, [0,1]]]) for name, features in node_data.items()}
        dst = {name: torch.stack([features[li,i] for li, i in pair_ids_list[:, [0,2]]]) for name, features in node_data.items()}
        # src = {name: torch.stack([nodes[i].data[name] for i in pair_ids[:, 0].tolist()]) if isinstance(node_data[name], torch.Tensor) else [nodes[i].data[name] for i in pair_ids[:, 0].tolist()] for name, features in node_data.items()}
        # dst = {name: torch.stack([nodes[i].data[name] for i in pair_ids[:, 1].tolist()]) if isinstance(node_data[name], torch.Tensor) else [nodes[i].data[name] for i in pair_ids[:, 1].tolist()] for name, features in node_data.items()}

        mess_res = processor.message_func(src, dst)

        for i, (li, parent_id, child_id) in enumerate(pair_ids_list):
            for name, value in mess_res.items():
                if nodes_list[li][child_id].mailbox[name].size(0) == 0:
                    nodes_list[li][child_id].mailbox[name] = value[i].unsqueeze(0)
                else:
                    nodes_list[li][child_id].mailbox[name] = torch.cat((nodes_list[li][child_id].mailbox[name], value[i].unsqueeze(0)), dim=0)
        
        child_ids_list, counts = torch.unique(pair_ids_list[:,[0,2]], dim=0, return_counts=True)
        max_mail_cnt = torch.max(counts)
        mailbox = {} # {name: torch.zeros(child_idx, max_mail_cnt, node.shape) for name in node_data.keys()}
        mask = {}
        for li, (nodes_id, child_id) in enumerate(child_ids_list):
            for name, value in nodes_list[nodes_id][child_id].mailbox.items():
                if name not in mailbox:
                    mailbox[name] = torch.zeros(child_ids_list.size(0), max_mail_cnt, value.size(-1), device=device)
                    mask[name] = torch.zeros(child_ids_list.size(0), max_mail_cnt, dtype=torch.bool, device=device)
                mailbox[name][li, :value.size(0)] = value.to(device)
                mask[name][li, :value.size(0)] = True

        reduce_res = processor.reduce_func(mailbox, mask)
        apply_res = processor.apply_node_func(reduce_res)

        for name, value in apply_res.items():
            for i, (nodes_id, child_id) in enumerate(child_ids_list):
                nodes_list[nodes_id][child_id].data[name] = value[i]
                result[nodes_id][name][child_id.item()] = value[i]
            
    return result

def topo_sort_levels(adj_matrix, reverse=False):
    """
    Topological Sort grouped by levels (hierarchies) for parallel processing.

    Args:
        adj_matrix (list of list of int): Adjacency matrix representing the DAG.
        reverse (bool): Whether to return levels in reverse order. Default is False.

    Returns:
        list of list: A list where each inner list represents a level (nodes that can be processed in parallel).

    Raises:
        ValueError: If the graph contains a cycle.
    """
    num_nodes = len(adj_matrix)
    in_degree = [0] * num_nodes  # Step 1: Compute in-degrees
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] == 1:
                in_degree[j] += 1

    # Step 2: Collect all nodes with in-degree of 0
    queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])
    levels = []  # This will store the nodes grouped by levels

    # Step 3: Process nodes level by level
    while queue:
        current_level = []  # Nodes at the current level
        for _ in range(len(queue)):  # Process all nodes in the current level
            current = queue.popleft()
            current_level.append(current)
            for neighbor in range(num_nodes):
                if adj_matrix[current][neighbor] == 1:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        levels.append(current_level)

    # Step 4: Check for remaining edges (detect cycles)
    if sum(in_degree) > 0:
        raise ValueError("The graph contains a cycle and cannot be topologically sorted.")

    # Step 5: Reverse levels if reverse=True
    return levels[::-1] if reverse else levels

def topo_sort_kahn(adj_matrix, reverse=False):
    """
    Topological Sort using Kahn's Algorithm with reverse option.

    Args:
        adj_matrix (list of list of int): Adjacency matrix representing the DAG.
        reverse (bool): Whether to return the reverse of the topological order. Default is False.

    Returns:
        list: A list representing the topological order of the nodes.

    Raises:
        ValueError: If the graph contains a cycle.
    """
    num_nodes = len(adj_matrix)
    in_degree = [0] * num_nodes  # Step 1: Compute in-degrees
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] == 1:
                in_degree[j] += 1

    # Step 2: Collect all nodes with in-degree of 0
    queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])
    topo_order = []

    # Step 3: Process nodes with Kahn's algorithm
    while queue:
        current = queue.popleft()
        topo_order.append(current)
        for neighbor in range(num_nodes):
            if adj_matrix[current][neighbor] == 1:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

    # Step 4: Check for remaining edges (detect cycles)
    if len(topo_order) != num_nodes:
        raise ValueError("The graph contains a cycle and cannot be topologically sorted.")

    # Step 5: Reverse the order if reverse=True
    return topo_order[::-1] if reverse else topo_order


def order_to_adj_matrix(order_tensor, mask_tensor=None):
    """
    Create an adjacency matrix from a parent-child relationship tensor, filtered by a mask tensor.
    
    Args:
        order_tensor (list of int): A list where each index represents a node, and the value at that index
                                    is the parent of that node (-1 if the node has no parent).
        mask_tensor (list of bool, optional): A list of booleans indicating which nodes to include in the adjacency matrix.

    Returns:
        list of list of int: Adjacency matrix representing the graph, filtered by the mask.
    """
    if mask_tensor is None:
        mask_tensor = [True] * len(order_tensor)  # If no mask provided, include all nodes

    # Create a mapping from original indices to filtered indices
    filtered_indices = [i for i, mask in enumerate(mask_tensor) if mask]
    index_map = {orig: new for new, orig in enumerate(filtered_indices)}
    num_nodes = len(filtered_indices)

    # Initialize the adjacency matrix with the correct size
    adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]

    # Populate the adjacency matrix using the mask
    for child, parent in enumerate(order_tensor):
        if mask_tensor[child] and parent != -1 and mask_tensor[parent]:
            adj_matrix[index_map[int(parent)]][index_map[int(child)]] = 1

    return torch.tensor(adj_matrix, dtype=torch.uint8)


class BaseProcessor:
    def message_func(self, src, dst):
        return {'x': src['x'], 'y': src['y']} # src, tgtのデータを使ってmessageを作成

    def reduce_func(self, node):
        x_sum = torch.sum(node.mailbox['x'], 0) # nodes.mailbox['x'] [num_nodes_with_same_level, num_neighbors, x_size]
        y_sum = torch.sum(node.mailbox['y'], 0)
        return {'xy': x_sum + y_sum} # reduceした結果を返す

    def apply_node_func(self, node):
        xy = node.data['xy']
        return {'x': xy, 'y': xy} # apply_node_funcでnodeの特徴量を更新

if __name__ == "__main__":
    # Example adjacency matrix (DAG)
    adj_matrix = [
        [0., 1., 1., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 1.],
        [0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.]
    ]
    adj_matrix = torch.tensor(adj_matrix)

    # Execute propagation
    processor = BaseProcessor()
    result = prop_nodes_topo(
        {
            'x': torch.arange(14).reshape(7, -1),
            'y': torch.arange(14).reshape(7, -1)
         },
        adj_matrix,
        processor,
        reverse=True,
    )

    print("Final aggregated messages:", result['x'][0])