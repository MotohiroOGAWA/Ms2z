from rdkit import Chem
import dill
from bidict import bidict
from collections import defaultdict
import copy
from tqdm import tqdm

from .utils import *
from .fragment_group import FragmentGroup, Fragment, FragBondList, FragmentBond, split_fragment_info


class FragmentNode:
    """
    Represents a node in the fragment tree.
    Each node has a depth, a dictionary of child nodes, and a list of leaves.
    """
    def __init__(self, depth, score):
        self.children = {}  # Dictionary of child nodes
        self.depth = depth  # Depth of the node in the tree
        self.leaves = []    # List of leaf values (associated fragments)
        self.score = score # Score of the node

    def __getstate__(self):
        state = self.__dict__.copy()
        state['children'] = {k: v for k, v in self.children.items()}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
    
    def add_child(self, keys:list, value):
        """
        Add a child to the current node recursively.
        :param keys: A list of keys representing the path to the value.
        :param value: The value to store at the leaf node.
        """
        key = keys[self.depth]
        score = self.score + sum(1 for c in key.split(',') if c not in ['-', '=', '#', ':'])
        if len(keys) == self.depth + 1:
            score += sum(1 for c in key if c in ['-', '=', '#', ':'])
        if key not in self.children:
            self.children[key] = FragmentNode(self.depth + 1, score)

        if self.depth < len(keys) - 1:
            self.children[key].add_child(keys, value)
        else:
            self.children[key].leaves.append(value)

    def to_dict(self):
        """
        Convert the FragmentNode and its children to a dictionary.
        :return: A dictionary representing the node and its children.
        """
        return {
            "depth": self.depth,
            "score": self.score,
            "leaves": self.leaves,
            "children": {key: child.to_dict() for key, child in self.children.items()}
        }

    @staticmethod
    def from_dict(data):
        """
        Create a FragmentNode from a dictionary representation.
        :param data: A dictionary containing the node data.
        :return: A FragmentNode object.
        """
        node = FragmentNode(data["depth"], data["score"])
        node.leaves = data["leaves"]
        node.children = {key: FragmentNode.from_dict(child_data) for key, child_data in data["children"].items()}
        return node

    def to_list(self, current_path=None):
        """
        Convert the FragmentNode tree into a list of (path, leaves, depth, score) tuples.
        :param current_path: Current path in the tree (used for recursion).
        :return: List of (path, leaves, depth, score) tuples.
        """
        if current_path is None:
            current_path = []
        
        result = []
        
        # Add the current node's leaves, depth, and score to the result
        if self.leaves:
            result.append((current_path, self.leaves, self.depth, self.score))
        
        # Recursively process children
        for key, child in self.children.items():
            result.extend(child.to_list(current_path + [key]))
        
        return result

    @staticmethod
    def from_list(data):
        """
        Reconstruct a FragmentNode tree from a list of (path, leaves, depth, score) tuples.
        :param data: List of (path, leaves, depth, score) tuples.
        :return: Reconstructed FragmentNode tree.
        """
        root = FragmentNode(depth=0, score=0)
        
        for path, leaves, depth, score in data:
            current_node = root
            for key in path:
                if key not in current_node.children:
                    current_node.children[key] = FragmentNode(depth=current_node.depth + 1, score=current_node.score)
                current_node = current_node.children[key]
            current_node.leaves.extend(leaves)
            current_node.depth = depth  # Ensure depth is restored
            current_node.score = score  # Ensure score is restored
        
        return root

    def display(self, level=0, prefix="", max_depth=None):
        """
        Recursively print the tree structure for visualization with tree-like characters.
        
        Args:
            level (int): Current indentation level for the node.
            prefix (str): Prefix string used for tree visualization.
            max_depth (int, optional): Maximum depth to display. If None, display all levels.
        """
        # Check if the current depth exceeds the max_depth
        if max_depth is not None and level > max_depth:
            return
        
        # Display the current node's leaves
        if self.leaves:
            print(f"{prefix}+- : {self.leaves}")

        # Display the children with tree structure
        for i, (key, child) in enumerate(self.children.items()):
            is_last = i == len(self.children) - 1
            connector = "`-" if is_last else "|-"
            child_prefix = prefix + ("   " if is_last else "|  ")
            print(f"{prefix}{connector} {key}")
            child.display(level + 1, child_prefix, max_depth=max_depth)

    def _write_to_file(self, file, prefix="", level=0, max_depth=None):
        """
        Recursively write the tree structure to an open file object with depth control.
        
        Args:
            file: An open file object to write the tree structure.
            prefix (str): Prefix string used for tree visualization.
            level (int): Current indentation level for the node.
            max_depth (int, optional): Maximum depth to write. If None, write all levels.
        """
        # Check if the current depth exceeds the max_depth
        if max_depth is not None and level > max_depth:
            return

        if self.leaves:
            file.write(f"{prefix}+- : {self.leaves}\n")

        for i, (key, child) in enumerate(self.children.items()):
            is_last = i == len(self.children) - 1
            connector = "`-" if is_last else "|-"
            child_prefix = prefix + ("   " if is_last else "|  ")
            file.write(f"{prefix}{connector} {key}\n")
            child._write_to_file(file, child_prefix, max_depth=max_depth)
    
class FragmentTree:
    """
    Represents the fragment tree structure, with methods for adding fragments,
    performing DFS traversal, and saving/loading the tree.
    """
    def __init__(self):
        self.root = FragmentNode(depth=0, score=0)
        self.smiles_and_atom_idx_to_potential = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['root'] = self.root 
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
    
    def search(self, ori_fragment: Fragment, bond_pos, start_atom_idx=None):
        ori_fragment = copy.deepcopy(ori_fragment)
        vocab_list = []
        visited = set()
        global_atom_map = bidict()
        if bond_pos == -1:
            root_next = [-1, '', (-1,-1)]
            bond_token = ''
        else:
            root_next = [-1, ori_fragment.bond_list[bond_pos].token, (-1,-1)]
            bond_token = ori_fragment.bond_list[bond_pos].token
        # mol = Chem.MolFromSmiles(smiles)
        # next_fragment_infoes = [{'frag': fragment_info, 'bond_pos': bond_pos, 'parent': (-1, -1), 'atom_map': list(range(mol.GetNumAtoms()))}]
        next_fragment_infoes = [{'frag': ori_fragment, 'bond_pos': bond_pos, 'parent': (-1, -1, bond_token)}]
        total_atom_nums = ori_fragment.mol.GetNumAtoms()
        current_atom_nums = 0
        while len(next_fragment_infoes) > 0:
            current_fragment_info_dict = next_fragment_infoes.pop(0)
            current_frag:Fragment = current_fragment_info_dict['frag']
            # current_fragment_info = current_fragment_info_dict['frag']
            # current_bond_infoes = list(zip(current_fragment_info[1::2], current_fragment_info[2::2]))
            current_bond_pos = current_fragment_info_dict['bond_pos']
            current_parent = current_fragment_info_dict['parent']
            if current_bond_pos == -1:
                start_global_atom_idx = current_frag.atom_map[start_atom_idx]
            else:
                start_global_atom_idx = current_frag.atom_map[current_frag.bond_list[current_bond_pos].atom_idx]
            # current_atom_map = current_fragment_info_dict['atom_map']
            # smiles = current_fragment_info[0]
            # mol = Chem.MolFromSmiles(smiles)

            results = []
            # if current_bond_pos == -1:
            if False:
                bond_tokens = []
                start_atom = current_frag.mol.GetAtomWithIdx(start_atom_idx)
                symbol = get_atom_symbol(start_atom)
                for neighbor in start_atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    bond = current_frag.mol.GetBondBetweenAtoms(start_atom_idx, neighbor_idx)
                    bond_tokens.append(chem_bond_to_token(bond.GetBondType()))
                for current_frag_bond in current_frag.bond_list:
                    current_frag_bond: FragmentBond
                    if start_atom_idx == current_frag_bond.atom_idx:
                        bond_tokens.append(current_frag_bond.token)
                bond_tokens = sorted(bond_tokens, key=lambda x: bond_priority[x])
                tree_keys = [symbol]
                if len(bond_tokens) > 1:
                    tree_keys.append(''.join(bond_tokens[1:]))
                
                current_node = self.root
                for i, key in enumerate(tree_keys):
                    if key in current_node.children:
                        current_node = current_node.children[key]
                        if (i == len(tree_keys) - 1 or i % 2 == 0) and len(current_node.leaves) > 0:
                            results.extend([(leave, current_node.score) for leave in current_node.leaves])
                    else:
                        break

            else:
                tree_keys = FragmentTree.get_tree_keys(current_frag.mol, start_atom_idx=current_frag.atom_map.index(start_global_atom_idx))
                current_node = self.root
                for i, key in enumerate(tree_keys):
                    if key in current_node.children:
                        current_node = current_node.children[key]
                        if (i == len(tree_keys) - 1 or i % 2 == 0) and len(current_node.leaves) > 0:
                            results.extend([(leave, current_node.score) for leave in current_node.leaves])
                    else:
                        break
            if len(results) == 0:
                raise ValueError(f'Not Found Atom Token: {current_frag.smiles}')
            results = sorted(results, key=lambda x: x[1], reverse=True) # [((smiles, start_atom_idx), score), ...]

            for result, _ in results:
                qry_frag_smiles = result[0]
                qry_frag_bond_list = FragBondList()
                qry_start_pos = result[1]
                qry_mol = Chem.MolFromSmiles(qry_frag_smiles)
                # qry_frag = Fragment(qry_frag_smiles, qry_frag_bond_list)
                # if current_bond_pos == -1:
                if False:
                    start_mol = Chem.MolFromSmiles(qry_frag_smiles)
                    qry_start_atom = start_mol.GetAtomWithIdx(0)
                    if get_atom_symbol(start_atom) != get_atom_symbol(qry_start_atom):
                        continue
                    if start_atom.GetNumExplicitHs() != qry_start_atom.GetNumExplicitHs():
                        continue
                    matched_q_to_t_global_atom_map = {start_atom_idx: start_atom_idx}
                else:
                    # matches = current_frag.GetSubstructMatches(qry_frag)
                    matches = current_frag.mol.GetSubstructMatches(qry_mol)
                    if len(matches) == 0:
                        continue
                    
                    start_atom_idx = current_frag.atom_map.index(start_global_atom_idx)
                    for match in matches:
                        if start_atom_idx in match:
                            break
                    else:
                        continue
                        # raise ValueError(f'Not Found Start Atom In Match: {current_frag}')
                    
                    match_with_global_atom_idx = [current_frag.atom_map[match_idx] for match_idx in match]
                    bond_arr = defaultdict(lambda: [0,0,0])
                    for atom in current_frag.mol_with_alt.GetAtoms():
                        if current_frag.atom_map_with_alt[atom.GetIdx()] not in match_with_global_atom_idx:
                            continue
                        global_atom_idx = current_frag.atom_map_with_alt[atom.GetIdx()]
                        bonds = atom.GetBonds()
                        for bond in bonds:
                            e_atom = bond.GetOtherAtom(atom)
                            if current_frag.atom_map_with_alt[e_atom.GetIdx()] in match_with_global_atom_idx:
                                continue
                            bond_type = bond.GetBondType()
                            bond_token = chem_bond_to_token(bond_type)
                            bond_num = token_to_num_bond(bond_token)
                            bond_arr[match_with_global_atom_idx.index(global_atom_idx)][bond_num-1] += 1

                    # Check if the potential is upper than the bonds in the current fragment
                    frag = True
                    pre_bond_list = []
                    for atom_idx, bond_info in bond_arr.items():
                        required_potential = bond_info[0] + bond_info[1]*2 + bond_info[2]*3
                        if required_potential > self.smiles_and_atom_idx_to_potential[qry_frag_smiles].get(atom_idx, -1):
                            frag = False
                            break
                        if bond_info[0] > 0:
                            pre_bond_list.extend([(atom_idx, '-')]*bond_info[0])
                        if bond_info[1] > 0:
                            pre_bond_list.extend([(atom_idx, '=')]*bond_info[1])
                        if bond_info[2] > 0:
                            pre_bond_list.extend([(atom_idx, '#')]*bond_info[2])
                    if not frag:
                        continue

                    bond_list = FragBondList(pre_bond_list)
                    qry_frag = Fragment(qry_frag_smiles, bond_list)

                    matched_q_to_t_global_atom_map = [current_frag.atom_map[match_idx] for match_idx in match]
                    matches2 = current_frag.GetSubstructMatches(qry_frag)
                    if len(matches2) == 0:
                        continue
                    for match in matches2:
                        if start_global_atom_idx in match:
                            break
                    else:
                        continue
                    matched_q_to_t_global_atom_map = [match_global_idx for match_global_idx in match]
                    break
            else:
                raise ValueError(f'Not Found Atom Token: {current_frag}')
            
            cut_remaining_atom_indices = set()
            # if current_bond_pos == -1:
            if False:
                cut_remaining_atom_indices.add(start_atom_idx)
                visited.update([current_frag.atom_map[start_atom_idx]])
            else:
                for qry_frag_bond in qry_frag.bond_list:
                    # if qry_start_pos == qry_frag_bond.id:
                    #     continue
                    cut_remaining_atom_indices.add(current_frag.atom_map.index(matched_q_to_t_global_atom_map[qry_frag_bond.atom_idx]))
                cut_remaining_atom_indices = sorted(list(cut_remaining_atom_indices))
                visited.update([v for v in matched_q_to_t_global_atom_map])

            cut_atom_pairs = []
            for cut_atom_idx in cut_remaining_atom_indices:
                atom = current_frag.mol.GetAtomWithIdx(cut_atom_idx)
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if current_frag.atom_map[neighbor_idx] not in visited:
                        cut_atom_pairs.append((cut_atom_idx, neighbor_idx))

            new_fragment_group = split_fragment_info(current_frag, cut_atom_pairs)

            if current_parent[0] == -1 and current_bond_pos == -1:
                root_next[2] = (0, -1)

            for new_fragment in new_fragment_group:
                new_fragment:Fragment
                if new_fragment == qry_frag:
                    next_parent_frag_idx = len(vocab_list)
                    if start_global_atom_idx not in new_fragment.atom_map:
                        continue
                    root_bond_poses = new_fragment.get_bond_poses(new_fragment.atom_map.index(start_global_atom_idx), current_parent[2])
                    
                    for s_bond_pos, (e_frag_idx, e_bond_pos) in new_fragment_group.get_neighbors(new_fragment.id).items():
                        next_fragment_infoes.append(
                            {'frag': new_fragment_group[e_frag_idx], 
                            'bond_pos': e_bond_pos, 
                            'parent': ( # (frag_idx, bond_pos, bond_token)
                                next_parent_frag_idx, s_bond_pos, 
                                new_fragment_group.get_bond_between(new_fragment.id, s_bond_pos, e_frag_idx, e_bond_pos).token),
                            'atom_map': new_fragment_group[e_frag_idx].atom_map
                            })
                        if s_bond_pos in root_bond_poses:
                            root_bond_poses.remove(s_bond_pos)

                    if current_bond_pos != -1 and current_bond_pos != 1 and len(root_bond_poses) == 0:
                        raise ValueError(f'Cannot Build Fragment Tree: {ori_fragment}')
                    if len(root_bond_poses) == 0:
                        root_bond_pos = -1
                    else:
                        root_bond_pos = root_bond_poses[0]

                    if current_parent[0] == -1:
                        root_next[2] = (0, root_bond_pos)
                    else:
                        vocab_list[current_parent[0]]['next'].append((current_parent[1], current_parent[2], (next_parent_frag_idx, root_bond_pos)))

                    for i, atom_i in enumerate(new_fragment.atom_map):
                        global_atom_map[atom_i] = (len(vocab_list), i)
                    vocab_list.append({'frag': qry_frag, 'idx': -1, 'next': []})
                    current_atom_nums += len(new_fragment.atom_map)
                    break

        if current_atom_nums != total_atom_nums:
            raise ValueError(f'Cannot Build Fragment Tree: {ori_fragment}')
        return tuple(root_next), vocab_list, global_atom_map
    
    def add_fragment(self, smi, atom_idx_to_potential):
        mol = Chem.MolFromSmiles(smi)
        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() > 0:
            return
        self.smiles_and_atom_idx_to_potential[smi] = dict(atom_idx_to_potential)
        for atom_idx, potential in atom_idx_to_potential.items():
            tree_keys = FragmentTree.get_tree_keys(mol, start_atom_idx=atom_idx)
            self.root.add_child(tree_keys, (smi, atom_idx))

    @staticmethod
    def get_tree_keys(fragment_mol, start_atom_idx):
        fragment_mol = copy.deepcopy(fragment_mol)
        traversal_order = []
        visited = set()
        FragmentTree.dfs(fragment_mol, visited, traversal_order, prev_atom_indices=[start_atom_idx])
        traversal_order = [sorted(v, key=lambda x: bond_priority.get(x, x)) for v in traversal_order]
        tree_keys = [','.join(map(str, v)) for v in traversal_order]
        return tree_keys

    @staticmethod
    def dfs(mol, visited, traversal_order, prev_atom_indices):
        next_atom_indices = []
        current_symbols = []
        current_bonds = []

        for atom_index in prev_atom_indices:
            if atom_index in visited:
                continue
            atom = mol.GetAtomWithIdx(atom_index)
            visited.add(atom_index)
            current_symbols.append(get_atom_symbol(atom))

            for neighbor in atom.GetNeighbors():
                neighbor_index = neighbor.GetIdx()
                if neighbor_index not in visited:
                    bond = mol.GetBondBetweenAtoms(atom_index, neighbor_index)
                    bond_type = bond.GetBondType()
                    bond_token = chem_bond_to_token(bond_type)
                    current_bonds.append(bond_token)
                    next_atom_indices.append(neighbor_index)

        next_atom_indices = list(set(next_atom_indices))
        if len(current_symbols) > 0:
            traversal_order.append(current_symbols)
        if len(current_bonds) > 0:
            traversal_order.append(current_bonds)

        if len(next_atom_indices) == 0:
            return traversal_order
        else:
            FragmentTree.dfs(mol, visited, traversal_order, next_atom_indices)

    # Example function to build the tree from a list of fragments
    def add_fragment_list(self, potentials:dict[str,dict[int,int]]):
        # Root node for the tree
        # root = FragmentNode('', 'Root')
        for smi, atom_idx_to_potential in tqdm(potentials.items(), mininterval=0.5, desc='Building Fragment Tree'):
            self.add_fragment(smi, atom_idx_to_potential)

    def save_tree(self, file_path):
        """
        Save the fragment tree to a file in binary format using dill.
        :param file_path: Path to the file where the tree will be saved.
        """
        with open(file_path, "wb") as file:
            dill.dump(self, file)

    @staticmethod
    def load_tree(file_path):
        """
        Load a fragment tree from a binary file.
        :param file_path: Path to the file from which the tree will be loaded.
        :return: The loaded FragmentTree object.
        """
        with open(file_path, "rb") as file:
            return dill.load(file)

    def to_dict(self):
        return self.root.to_dict()

    @staticmethod
    def from_dict(data):
        tree = FragmentTree()
        tree.root = FragmentNode.from_dict(data)
        return tree

    def to_list(self):
        return self.root.to_list()
    
    @staticmethod
    def from_list(data):
        tree = FragmentTree()
        tree.root = FragmentNode.from_list(data)
        return tree

    def display_tree(self, max_depth=None):
        """
        Display the entire tree structure for visualization.
        """
        self.root.display(max_depth=max_depth)

    def save_to_file_incrementally(self, file_path):
        """
        Save the tree structure to a file incrementally while keeping the file open.
        :param file_path: Path to the file where the tree will be saved.
        """
        with open(file_path, "w") as file:
            self.root._write_to_file(file)

# def build_route_tree(frag_info, start_bond_pos, mol=None, options=None):
#     if mol is None:
#         mol = Chem.MolFromSmiles(frag_info[0])
    
#     if options is not None and 'max_route' in options:
#         max_route = options['max_route']
#     else:
#         max_route = float('inf')
        
#     bond_infoes = [(bond_idx, bond_type) for bond_idx, bond_type in zip(frag_info[1::2], frag_info[2::2])]
#     frag_info_dict = defaultdict(list)
#     for i in range(len(bond_infoes)):
#         frag_info_dict[bond_infoes[i][0]].append(i)
    
#     visited = set()
#     completed_routes = []
#     start_atom_idx = bond_infoes[start_bond_pos][0]
#     start_atom = mol.GetAtomWithIdx(start_atom_idx)
#     current_routes = [{'idx': [start_atom_idx], 'route': []}]
#     current_routes[0]['route'].append(bond_infoes[start_bond_pos][1])
#     current_routes[0]['route'].append(get_atom_symbol(start_atom))
#     visited.add(bond_infoes[start_bond_pos][0])
#     for i, bond_info in enumerate(bond_infoes):
#         if i == start_bond_pos:
#             continue
#         if bond_info[0] == start_atom_idx:
#             route = copy.deepcopy(current_routes[0])
#             route['route'].append(bond_info[1])
#             completed_routes.append(route)

#     if len(visited) == mol.GetNumAtoms():
#         if len(completed_routes) == 0: # -O などの1つの原子で続きの結合がない場合 
#             completed_routes.append(current_routes[0])
#         next_routes = []
#         current_routes = []
    
#     route_cnt = 1
#     while len(current_routes) > 0:
#         next_routes = []
#         for i, current_route in enumerate(reversed(current_routes)):
#             current_atom = mol.GetAtomWithIdx(current_route['idx'][-1])
#             neighbors = [neighbor for neighbor in current_atom.GetNeighbors() if neighbor.GetIdx() not in visited]

#             if len(neighbors) == 0:
#                 if len(current_route['route']) % 2 == 0: # -C-C などの続きの結合がない場合
#                     completed_routes.append(current_route)
#                 continue

#             for neighbor in neighbors:
#                 neighbor_idx = neighbor.GetIdx()
#                 visited.add(neighbor_idx)
#                 new_route = copy.deepcopy(current_route)
#                 bond = mol.GetBondBetweenAtoms(current_route['idx'][-1], neighbor_idx)
#                 bond_type = chem_bond_to_token(bond.GetBondType())
#                 new_route['route'].append(bond_type)
#                 if route_cnt < max_route:
#                     new_route['idx'].append(neighbor_idx)
#                     new_route['route'].append(get_atom_symbol(neighbor))
                
#                     for i, bond_info in enumerate(bond_infoes):
#                         if neighbor_idx != bond_info[0]:
#                             continue
#                         route = new_route.copy()
#                         route['route'].append(bond_info[1])
#                         completed_routes.append(route)
                
#                 next_routes.append(new_route)

#         current_routes = next_routes
#         route_cnt += 1
#         if route_cnt > max_route:
#             break

    
#     for current_route in current_routes:
#         completed_routes.append(current_route)
    
#     return completed_routes
