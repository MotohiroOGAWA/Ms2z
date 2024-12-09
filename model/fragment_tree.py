from rdkit import Chem
import dill
from bidict import bidict
from collections import defaultdict
import copy
from tqdm import tqdm

from .utils import *
from .fragment import Fragment
from .fragment_bond import FragBondList, FragmentBond


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
    
    def add_child(self, keys:list, value):
        """
        Add a child to the current node recursively.
        :param keys: A list of keys representing the path to the value.
        :param value: The value to store at the leaf node.
        """
        key = keys[self.depth]
        score = self.score + sum(1 for c in key if c.isdigit())
        if len(keys) == self.depth + 1:
            score += sum(1 for c in key if c in ['-', '=', '#', ':'])
        if key not in self.children:
            self.children[key] = FragmentNode(self.depth + 1, score)

        if self.depth < len(keys) - 1:
            self.children[key].add_child(keys, value)
        else:
            self.children[key].leaves.append(value)

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
    
    def search(self, fragment_info, bond_pos, start_atom_idx=None):
        vocab_list = []
        visited = set()
        global_atom_map = bidict()
        if bond_pos == -1:
            root_next = [-1, '', (-1,-1)]
        else:
            root_next = [-1, fragment_info[2*bond_pos+2], (-1,-1)]
        smiles = fragment_info[0]
        mol = Chem.MolFromSmiles(smiles)
        total_atom_nums = mol.GetNumAtoms()
        current_atom_nums = 0
        next_fragment_infoes = [{'frag': fragment_info, 'bond_pos': bond_pos, 'parent': (-1, -1), 'atom_map': list(range(mol.GetNumAtoms()))}]
        while len(next_fragment_infoes) > 0:
            current_fragment_info_dict = next_fragment_infoes.pop(0)
            current_fragment_info = current_fragment_info_dict['frag']
            current_bond_infoes = list(zip(current_fragment_info[1::2], current_fragment_info[2::2]))
            current_bond_pos = current_fragment_info_dict['bond_pos']
            current_parent = current_fragment_info_dict['parent']
            current_atom_map = current_fragment_info_dict['atom_map']
            smiles = current_fragment_info[0]
            mol = Chem.MolFromSmiles(smiles)

            results = []
            if current_bond_pos == -1:
                bond_types = []
                start_atom = mol.GetAtomWithIdx(start_atom_idx)
                symbol = str(start_atom.GetAtomicNum())
                for neighbor in start_atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    bond = mol.GetBondBetweenAtoms(start_atom_idx, neighbor_idx)
                    bond_types.append(chem_bond_to_token(bond.GetBondType()))
                for current_bond_info in current_bond_infoes:
                    if start_atom_idx == current_bond_info[0]:
                        bond_types.append(current_bond_info[1])
                bond_types = sorted(bond_types, key=lambda x: bond_priority[x])
                tree_keys = [bond_types[0], symbol]
                if len(bond_types) > 1:
                    tree_keys.append(''.join(bond_types[1:]))
                
                current_node = self.root
                for i, key in enumerate(tree_keys):
                    if key in current_node.children:
                        current_node = current_node.children[key]
                        if (i == len(tree_keys) - 1 or i % 2 == 0) and len(current_node.leaves) > 0:
                            results.extend([(leave, current_node.score) for leave in current_node.leaves])
                    else:
                        break

            else:
                tree_keys = FragmentTree.get_tree_keys(Chem.MolFromSmiles(smiles), current_bond_infoes, start_bond_pos=current_bond_pos)
                current_node = self.root
                for i, key in enumerate(tree_keys):
                    if key in current_node.children:
                        current_node = current_node.children[key]
                        if (i == len(tree_keys) - 1 or i % 2 == 0) and len(current_node.leaves) > 0:
                            results.extend([(leave, current_node.score) for leave in current_node.leaves])
                    else:
                        break
            if len(results) == 0:
                raise ValueError(f'Not Found Atom Token: {current_fragment_info}')
            results = sorted(results, key=lambda x: x[1], reverse=True)

            for result, _ in results:
                qry_frag_info = result[0]
                qry_start_pos = result[1]
                if current_bond_pos == -1:
                    start_mol = Chem.MolFromSmiles(qry_frag_info[0])
                    qry_start_atom = start_mol.GetAtomWithIdx(0)
                    if start_atom.GetAtomicNum() != qry_start_atom.GetAtomicNum():
                        continue
                    if start_atom.GetNumExplicitHs() != qry_start_atom.GetNumExplicitHs():
                        continue
                    matched_q_to_t_atom_map = {start_atom_idx: start_atom_idx}
                else:
                    matched_q_to_t_atom_map = match_fragment(tgt_frag_info=current_fragment_info, tgt_start_pos=current_bond_pos, qry_frag_info=qry_frag_info, qry_start_pos=qry_start_pos)
                if matched_q_to_t_atom_map is not None:
                    break
            else:
                raise ValueError(f'Not Found Atom Token: {current_fragment_info}')
            
            cut_remaining_atom_indices = set()
            if current_bond_pos == -1:
                cut_remaining_atom_indices.add(start_atom_idx)
                visited.update([start_atom_idx])
            else:
                flag = False
                for bond_idx, bond_type in zip(qry_frag_info[1::2], qry_frag_info[2::2]):
                    if not flag and (bond_idx == qry_frag_info[2 * qry_start_pos + 1]) and (bond_type == qry_frag_info[2 * qry_start_pos + 2]):
                        flag = True
                        continue
                    cut_remaining_atom_indices.add(matched_q_to_t_atom_map[bond_idx])
                cut_remaining_atom_indices = sorted(list(cut_remaining_atom_indices))
                visited.update([current_atom_map[v] for v in matched_q_to_t_atom_map.values()])

            cut_atom_pairs = []
            for cut_atom_idx in cut_remaining_atom_indices:
                atom = mol.GetAtomWithIdx(cut_atom_idx)
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if current_atom_map[neighbor_idx] not in visited:
                        cut_atom_pairs.append((cut_atom_idx, neighbor_idx))

            new_fragment_infoes, atom_map = split_fragment_info(current_fragment_info, cut_atom_pairs, current_atom_map)

            if current_parent[0] == -1 and current_bond_pos == -1:
                root_next[2] = (0, -1)

            for frag_idx, new_fragment_info_with_joint in enumerate(new_fragment_infoes):
                new_fragment_info = new_fragment_info_with_joint[0]
                if new_fragment_info == qry_frag_info:
                    joint_info = new_fragment_info_with_joint[1]
                    # joint_dict = {bond_idx: (next_frag_idx, next_bond_pos)}
                    joint_dict = {bond_idx: value for bond_idx, value in joint_info}
                    next_parent_frag_idx = len(vocab_list)
                    
                    parent_flag = False
                    for cur_bond_pos, joint_info in enumerate(zip(new_fragment_info[1::2], new_fragment_info[2::2])):
                        if cur_bond_pos in joint_dict:
                            mappping = [k for k, v in sorted(atom_map.items(), key=lambda x: x[1][1]) if v[0] == joint_dict[cur_bond_pos][0]]
                            
                            next_fragment_infoes.append(
                                {'frag': new_fragment_infoes[joint_dict[cur_bond_pos][0]][0], 
                                 'bond_pos': joint_dict[cur_bond_pos][1], 
                                 'parent': (next_parent_frag_idx, cur_bond_pos, new_fragment_infoes[joint_dict[cur_bond_pos][0]][0][2*joint_dict[cur_bond_pos][1]+2]),
                                 'atom_map': mappping
                                 })
                        else: # グループ内のフラグメントに接続する結合でない場合
                            if (current_parent[0] == -1 
                                and current_bond_pos != -1 
                                and current_bond_infoes[current_bond_pos][1] == joint_info[1]):
                                root_next[2] = (0, cur_bond_pos) # このフラグメントグループの初めの結合部分 (frag_id, bond_pos)
                            else:
                                if len(joint_dict) == 0: # 切断されず次のフラグメントがない場合
                                    atom_idx = current_fragment_info[1::2][current_bond_pos]
                                    for k, (atom_idx2, bond_type) in enumerate(zip(new_fragment_info[1::2], new_fragment_info[2::2])):
                                        if atom_idx2 == atom_idx and current_parent[2] == bond_type:
                                            vocab_list[current_parent[0]]['next'].append((current_parent[1], current_parent[2], (next_parent_frag_idx, k)))
                                            break
                                    break
                                elif current_parent[0] != -1:
                                    if not parent_flag:
                                        parent_flag = True
                                        vocab_list[current_parent[0]]['next'].append((current_parent[1], current_parent[2], (next_parent_frag_idx, cur_bond_pos)))
                        
                    mappping = [k for k, v in sorted(atom_map.items(), key=lambda x: x[1][1]) if v[0] == frag_idx]
                    for i, atom_i in enumerate(mappping):
                        global_atom_map[atom_i] = (len(vocab_list), i)
                    vocab_list.append({'frag': qry_frag_info, 'idx': -1, 'next': []})
                    current_atom_nums += len(mappping)
                    break
        if current_atom_nums != total_atom_nums:
            error_message =  '\n'.join([f'{k}: {v}' for k, v in enumerate(vocab_list)])
            raise ValueError(f'Cannot Build Fragment Tree: {fragment_info}\n{error_message}')
        return tuple(root_next), vocab_list, global_atom_map
    
    def add_fragment(self, fragment, fragment_mol):
        """
        Add a fragment to the tree based on the molecular structure and binding sites.
        :param fragment: A tuple representing the fragment structure.
        :param fragment_mol: The RDKit molecule object for the fragment.
        """
        binding_sites_len = (len(fragment) - 1) // 2
        binding_sites = list(zip(fragment[1::2], fragment[2::2]))

        for i, binding_site in enumerate(binding_sites):
            tree_keys = FragmentTree.get_tree_keys(fragment_mol, binding_sites, start_bond_pos=i)
            self.root.add_child(tree_keys, (fragment, i))

    @staticmethod
    def get_tree_keys(fragment_mol, binding_sites, start_bond_pos):
        """
        Generate a list of keys for the tree structure based on the fragment structure.
        :param fragment_mol: The RDKit molecule object for the fragment.
        :param binding_sites: List of binding sites in the fragment.
        :param start_bond_pos: The starting position for the bond traversal.
        :return: A list of keys representing the tree structure.
        """
        traversal_order = [[binding_sites[start_bond_pos][1]]]
        visited = set()
        FragmentTree.dfs(fragment_mol, visited, traversal_order, prev_atom_indices=[binding_sites[start_bond_pos][0]], bonding_sites=[bonding_site for i, bonding_site in enumerate(binding_sites) if i != start_bond_pos])
        traversal_order = [sorted(v, key=lambda x: bond_priority.get(x, x)) for v in traversal_order]
        tree_keys = [','.join(map(str, v)) for v in traversal_order]
        return tree_keys

    @staticmethod
    def dfs(mol, visited, traversal_order, prev_atom_indices, bonding_sites):
        """
        Perform a depth-first search traversal of the molecular structure to generate tree keys.
        :param mol: The RDKit molecule object for the fragment.
        :param visited: Set of visited atom indices.
        :param traversal_order: List to store the traversal order.
        :param prev_atom_indices: List of previous atom indices.
        :param bonding_sites: List of binding sites in the fragment.
        """
        next_atom_indices = []
        current_symbols = []
        current_bonds = []

        for atom_index in prev_atom_indices:
            if atom_index in visited:
                continue
            atom = mol.GetAtomWithIdx(atom_index)
            visited.add(atom_index)
            current_symbols.append(atom.GetAtomicNum())
            for bonding_site in bonding_sites:
                if bonding_site[0] == atom_index:
                    current_bonds.append(bonding_site[1])

            for neighbor in atom.GetNeighbors():
                neighbor_index = neighbor.GetIdx()
                if neighbor_index not in visited:
                    bond = mol.GetBondBetweenAtoms(atom_index, neighbor_index)
                    bond_type = bond.GetBondType()
                    bonding_type = chem_bond_to_token(bond_type)
                    current_bonds.append(bonding_type)
                    next_atom_indices.append(neighbor_index)

        next_atom_indices = list(set(next_atom_indices))
        if len(current_symbols) > 0:
            traversal_order.append(current_symbols)
        if len(current_bonds) > 0:
            traversal_order.append(current_bonds)

        if len(next_atom_indices) == 0:
            return traversal_order
        else:
            FragmentTree.dfs(mol, visited, traversal_order, next_atom_indices, bonding_sites)

    # Example function to build the tree from a list of fragments
    def add_fragment_list(self, fragments, fragment_mols=None):
        # Root node for the tree
        # root = FragmentNode('', 'Root')
        for i, fragment in tqdm(enumerate(fragments), total=len(fragments), mininterval=0.5, desc='Building Fragment Tree'):
            if fragment_mols is None:
                fragment_mol = Chem.MolFromSmiles(fragment[0])
            else:
                fragment_mol = fragment_mols[i]
            self.add_fragment(fragment, fragment_mol)

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


def match_fragment(tgt_frag_info, tgt_start_pos, qry_frag_info, qry_start_pos):
    tgt_mol = Chem.MolFromSmiles(tgt_frag_info[0])
    qry_mol = Chem.MolFromSmiles(qry_frag_info[0])
    tgt_bond_infoes = [(bond_idx, bond_type) for bond_idx, bond_type in zip(tgt_frag_info[1::2], tgt_frag_info[2::2])]

    qry_bond_infoes = [(bond_idx, bond_type) for bond_idx, bond_type in zip(qry_frag_info[1::2], qry_frag_info[2::2])]


    if tgt_bond_infoes[tgt_start_pos][1] != qry_bond_infoes[qry_start_pos][1]:
        return None

    qry_route_trees = build_route_tree(qry_frag_info, qry_start_pos, mol=qry_mol)
    max_route = max([len(x['idx']) for x in qry_route_trees])
    tgt_route_trees = build_route_tree(tgt_frag_info, tgt_start_pos, mol=tgt_mol, options={'max_route': max_route})

    qry_route_trees = sorted(qry_route_trees, key = lambda x: "_".join(x['route']), reverse = True)
    qry_route_strs = ["_".join(x['route']) for x in qry_route_trees]
    tgt_route_trees = sorted(tgt_route_trees, key = lambda x: "_".join(x['route']), reverse = True)
    tgt_route_strs = ["_".join(x['route']) for x in tgt_route_trees]
    
    route_pair = defaultdict(list)
    nj = 0
    for i, qry_route_tree in enumerate(qry_route_trees):
        j = nj
        while j < len(tgt_route_trees):
            if qry_route_strs[i] > tgt_route_strs[j]:
                nj = j
                break
            if qry_route_strs[i] == tgt_route_strs[j]:
                route_pair[i].append(j)
            j += 1
    
    if len(qry_route_strs) != len(route_pair):
        return None
    
    atom_idx_dicts = []
    for qry_route_idx, tgt_route_indices in route_pair.items():
        qry_route_tree = qry_route_trees[qry_route_idx]
        if len(atom_idx_dicts) == 0:
            for tgt_route_idx in tgt_route_indices:
                atom_idx_dict = {qry_atom_idx: tgt_atom_idx for qry_atom_idx, tgt_atom_idx in zip(qry_route_tree['idx'], tgt_route_trees[tgt_route_idx]['idx'])}
                atom_idx_dicts.append(atom_idx_dict)
        else:
            new_atom_idx_dicts = []
            for atom_idx_dict in atom_idx_dicts:
                for tgt_route_idx in tgt_route_indices:
                    tmp_atom_idx_dict = copy.deepcopy(atom_idx_dict)
                    for qry_atom_idx, tgt_atom_idx in zip(qry_route_tree['idx'], tgt_route_trees[tgt_route_idx]['idx']):
                        if (qry_atom_idx in tmp_atom_idx_dict) and tmp_atom_idx_dict[qry_atom_idx] != tgt_atom_idx:
                            break
                        if tgt_atom_idx in tmp_atom_idx_dict.values():
                            continue
                        tmp_atom_idx_dict[qry_atom_idx] = tgt_atom_idx
                    else:
                        new_atom_idx_dicts.append(tmp_atom_idx_dict)
            atom_idx_dicts = new_atom_idx_dicts

        if len(atom_idx_dicts) == 0:
            break
    
    atom_idx_dicts = [atom_idx_dict for atom_idx_dict in atom_idx_dicts if len(atom_idx_dict) == len(qry_mol.GetAtoms())]
    if len(atom_idx_dicts) == 0:
        return None

    return atom_idx_dicts[0]


def build_route_tree(frag_info, start_bond_pos, mol=None, options=None):
    if mol is None:
        mol = Chem.MolFromSmiles(frag_info[0])
    
    if options is not None and 'max_route' in options:
        max_route = options['max_route']
    else:
        max_route = float('inf')
        
    bond_infoes = [(bond_idx, bond_type) for bond_idx, bond_type in zip(frag_info[1::2], frag_info[2::2])]
    frag_info_dict = defaultdict(list)
    for i in range(len(bond_infoes)):
        frag_info_dict[bond_infoes[i][0]].append(i)
    
    visited = set()
    completed_routes = []
    start_atom_idx = bond_infoes[start_bond_pos][0]
    start_atom = mol.GetAtomWithIdx(start_atom_idx)
    current_routes = [{'idx': [start_atom_idx], 'route': []}]
    current_routes[0]['route'].append(bond_infoes[start_bond_pos][1])
    current_routes[0]['route'].append(get_atom_symbol(start_atom))
    visited.add(bond_infoes[start_bond_pos][0])
    for i, bond_info in enumerate(bond_infoes):
        if i == start_bond_pos:
            continue
        if bond_info[0] == start_atom_idx:
            route = copy.deepcopy(current_routes[0])
            route['route'].append(bond_info[1])
            completed_routes.append(route)

    if len(visited) == mol.GetNumAtoms():
        if len(completed_routes) == 0: # -O などの1つの原子で続きの結合がない場合 
            completed_routes.append(current_routes[0])
        next_routes = []
        current_routes = []
    
    route_cnt = 1
    while len(current_routes) > 0:
        next_routes = []
        for i, current_route in enumerate(reversed(current_routes)):
            current_atom = mol.GetAtomWithIdx(current_route['idx'][-1])
            neighbors = [neighbor for neighbor in current_atom.GetNeighbors() if neighbor.GetIdx() not in visited]

            if len(neighbors) == 0:
                if len(current_route['route']) % 2 == 0: # -C-C などの続きの結合がない場合
                    completed_routes.append(current_route)
                continue

            for neighbor in neighbors:
                neighbor_idx = neighbor.GetIdx()
                visited.add(neighbor_idx)
                new_route = copy.deepcopy(current_route)
                bond = mol.GetBondBetweenAtoms(current_route['idx'][-1], neighbor_idx)
                bond_type = chem_bond_to_token(bond.GetBondType())
                new_route['route'].append(bond_type)
                if route_cnt < max_route:
                    new_route['idx'].append(neighbor_idx)
                    new_route['route'].append(get_atom_symbol(neighbor))
                
                    for i, bond_info in enumerate(bond_infoes):
                        if neighbor_idx != bond_info[0]:
                            continue
                        route = new_route.copy()
                        route['route'].append(bond_info[1])
                        completed_routes.append(route)
                
                next_routes.append(new_route)

        current_routes = next_routes
        route_cnt += 1
        if route_cnt > max_route:
            break

    
    for current_route in current_routes:
        completed_routes.append(current_route)
    
    return completed_routes


def split_fragment_info(frag_info, cut_bond_indices, ori_atom_indices):
    """
    Break bonds at specified atom indices and generate new frag_info for resulting fragments.
    
    Parameters:
        frag_info (tuple): The original fragment information, e.g., ('CC(C)CCCCCCOC(=O)', 0, '-', 10, '-').
        bond_indices (list of tuples): Pairs of atom indices to break bonds (e.g., [(1, 3)]).
    
    Returns:
        list of tuples: New frag_info for each resulting fragment.
    """
    # Extract the SMILES string from frag_info
    smiles = frag_info[0]

    # Convert SMILES to Mol object
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string in frag_info.")
    mol = Chem.rdmolops.RemoveHs(mol)
    
    # Create editable version of the molecule
    new_mol = Chem.RWMol(mol)

    for atom in new_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    
    # Break specified bonds
    bond_types = {}
    for idx1, idx2 in cut_bond_indices:
        if new_mol.GetBondBetweenAtoms(idx1, idx2) is not None:
            atom1 = new_mol.GetAtomWithIdx(idx1)
            atom2 = new_mol.GetAtomWithIdx(idx2)
            bond = new_mol.GetBondBetweenAtoms(idx1, idx2)
            new_mol = add_Hs(new_mol, atom1, atom2, bond)  # Add hydrogens to adjust valence
            new_mol.RemoveBond(atom1.GetIdx(), atom2.GetIdx())
            if idx1 < idx2:
                bond_types[(idx1, idx2)] = chem_bond_to_token(bond.GetBondType())
            else:
                bond_types[(idx2, idx1)] = chem_bond_to_token(bond.GetBondType())
        else:
            raise ValueError(f"No bond found between atom indices {idx1} and {idx2}.")
    
    # Generate new fragment information for each resulting fragment
    new_mol = new_mol.GetMol()
    new_mol = sanitize(new_mol, kekulize = False)
    new_smiles = Chem.MolToSmiles(new_mol)
    fragment_mols = [Chem.MolFromSmiles(fragment) for fragment in new_smiles.split('.')]
    fragment_mols = [sanitize(fragment, kekulize = False) for fragment in fragment_mols]

    frag_infoes = []
    atom_dicts = {}
    bond_poses_dict = defaultdict(lambda: defaultdict(list))
    for frag_idx, frag_mol in enumerate(fragment_mols):
        
        atom_dict = {}
        for i, atom in enumerate(frag_mol.GetAtoms()):
            amap = atom.GetAtomMapNum()
            atom_dict[amap] = i
            
        for atom in frag_mol.GetAtoms():
            frag_mol.GetAtomWithIdx(atom.GetIdx()).SetAtomMapNum(0)

        frag_smi = Chem.MolToSmiles(frag_mol)
        atom_order = list(map(int, frag_mol.GetProp('_smilesAtomOutputOrder')[1:-2].split(",")))

        for key in atom_dict:
            atom_dict[key] = atom_order.index(atom_dict[key])
        
        bond_infoes = []
        for i in range(1, len(frag_info), 2):
            if frag_info[i] in atom_dict:
                bond_infoes.append([atom_dict[frag_info[i]], frag_info[i+1]])
        for (idx1, idx2) in cut_bond_indices:
            if idx1 in atom_dict:
                idx = idx1
            elif idx2 in atom_dict:
                idx = idx2
            else:
                continue
            
            bond_type = bond_types.get((idx1, idx2), bond_types.get((idx2, idx1)))
            bond_infoes.append([atom_dict[idx], bond_type])
        bond_infoes = sorted(bond_infoes, key = lambda x: x[0])

        new_frag_info = [frag_smi]
        for i in range(len(bond_infoes)):
            new_frag_info.append(bond_infoes[i][0])
            new_frag_info.append(bond_infoes[i][1])
            bond_poses_dict[frag_idx][(bond_infoes[i][0], bond_infoes[i][1])].append(i)
        frag_infoes.append([tuple(new_frag_info), []])
        atom_dicts[frag_idx] = atom_dict
    
    for (bond_idx1, bond_idx2) in cut_bond_indices:
        for frag_idx, atom_dict in atom_dicts.items():
            if bond_idx1 in atom_dict:
                bond_i1 = atom_dict[bond_idx1]
                frag_idx1 = frag_idx
            if bond_idx2 in atom_dict:
                bond_i2 = atom_dict[bond_idx2]
                frag_idx2 = frag_idx
        bond = mol.GetBondBetweenAtoms(bond_idx1, bond_idx2)
        bond_type = chem_bond_to_token(bond.GetBondType())
        frag_infoes[frag_idx1][1].append((bond_poses_dict[frag_idx1][(bond_i1, bond_type)][0], (frag_idx2, bond_poses_dict[frag_idx2][(bond_i2, bond_type)][0])))
        frag_infoes[frag_idx2][1].append((bond_poses_dict[frag_idx2][(bond_i2, bond_type)].pop(0), (frag_idx1, bond_poses_dict[frag_idx1][(bond_i1, bond_type)].pop(0))))
    
    for i in range(len(frag_infoes)):
        frag_infoes[i][1] = sorted(frag_infoes[i][1], key = lambda x: x[0])

    normalize_bond_dict = defaultdict(dict)
    normalize_frag_infoes = []
    final_atom_maps = {}
    for frag_idx, frag_mol in enumerate(fragment_mols):
        frag_info = frag_infoes[frag_idx][0]
        frag_bond_list = FragBondList([FragmentBond(a_idx, bond_token) for a_idx, bond_token in zip(frag_info[1::2], frag_info[2::2])])
        fragment = Fragment(frag_info[0], frag_bond_list)
        new_frag_info = tuple(fragment)
        atom_maps = fragment.atom_map
        normalize_frag_infoes.append([new_frag_info, []])
        for global_atom_i, pre_frag_atom_i in atom_dicts[frag_idx].items():
            final_atom_maps[ori_atom_indices[global_atom_i]] = (frag_idx, atom_maps.index(pre_frag_atom_i))
        new_bond_dict = defaultdict(list)
        for i, (new_atom_idx, bond_type) in enumerate(zip(new_frag_info[1::2], new_frag_info[2::2])):
            new_bond_dict[(new_atom_idx, bond_type)].append(i)
        for i, pre_atom_idx in enumerate(frag_info[1::2]):
            normalize_bond_dict[frag_idx][i] = new_bond_dict[(atom_maps.index(pre_atom_idx), frag_info[2*i+2])].pop(0)

    for frag_idx, frag_info in enumerate(frag_infoes):
        for bond_pos, (to_frag_idx, to_bond_pos) in frag_info[1]:
            normalize_frag_infoes[frag_idx][1].append((normalize_bond_dict[frag_idx][bond_pos], (to_frag_idx, normalize_bond_dict[to_frag_idx][to_bond_pos])))
        
    return normalize_frag_infoes, final_atom_maps