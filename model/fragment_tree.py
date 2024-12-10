from rdkit import Chem
import dill
from bidict import bidict
from collections import defaultdict
import copy
from tqdm import tqdm

from .utils import *
from .fragment_group import FragmentGroup, Fragment, FragBondList, FragmentBond


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
                start_global_atom_idx = start_atom_idx
            else:
                start_global_atom_idx = current_frag.atom_map[current_frag.bond_list[current_bond_pos].atom_idx]
            # current_atom_map = current_fragment_info_dict['atom_map']
            # smiles = current_fragment_info[0]
            # mol = Chem.MolFromSmiles(smiles)

            results = []
            if current_bond_pos == -1:
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
                tree_keys = [bond_tokens[0], symbol]
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
                tree_keys = FragmentTree.get_tree_keys(copy.deepcopy(current_frag.mol), current_frag.bond_list, start_bond_pos=current_bond_pos)
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
            results = sorted(results, key=lambda x: x[1], reverse=True)

            for result, _ in results:
                qry_frag_smiles = result[0]
                qry_frag_bond_list = FragBondList(result[1])
                qry_start_pos = result[2]
                qry_frag = Fragment(qry_frag_smiles, qry_frag_bond_list)
                if current_bond_pos == -1:
                    start_mol = Chem.MolFromSmiles(qry_frag_smiles)
                    qry_start_atom = start_mol.GetAtomWithIdx(0)
                    if start_atom.GetAtomicNum() != qry_start_atom.GetAtomicNum():
                        continue
                    if start_atom.GetNumExplicitHs() != qry_start_atom.GetNumExplicitHs():
                        continue
                    matched_q_to_t_atom_map = {start_atom_idx: start_atom_idx}
                else:
                    matches = current_frag.GetSubstructMatches(qry_frag)
                    if len(matches) == 0:
                        continue
                    matched_q_to_t_atom_map = None
                    for match in matches:
                        if current_frag.atom_map[current_frag.bond_list[current_bond_pos].atom_idx] in match:
                            matched_q_to_t_atom_map = match
                            break
                if matched_q_to_t_atom_map is not None:
                    break
            else:
                raise ValueError(f'Not Found Atom Token: {current_frag}')
            
            cut_remaining_atom_indices = set()
            if current_bond_pos == -1:
                cut_remaining_atom_indices.add(start_atom_idx)
                visited.update([start_atom_idx])
            else:
                for qry_frag_bond in qry_frag.bond_list:
                    # if qry_start_pos == qry_frag_bond.id:
                    #     continue
                    cut_remaining_atom_indices.add(current_frag.atom_map.index(matched_q_to_t_atom_map[qry_frag_bond.atom_idx]))
                cut_remaining_atom_indices = sorted(list(cut_remaining_atom_indices))
                visited.update([v for v in matched_q_to_t_atom_map])

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

                    if current_bond_pos != 1 and len(root_bond_poses) == 0:
                        raise ValueError(f'Cannot Build Fragment Tree: {ori_fragment}')
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
            error_message =  '\n'.join([f'{k}: {v}' for k, v in enumerate(vocab_list)])
            raise ValueError(f'Cannot Build Fragment Tree: {ori_fragment}\n{error_message}')
        return tuple(root_next), vocab_list, global_atom_map
    
    def add_fragment(self, fragment: Fragment):
        for frag_bond in fragment.bond_list:
            tree_keys = FragmentTree.get_tree_keys(fragment.mol, fragment.bond_list, start_bond_pos=frag_bond.id)
            self.root.add_child(tree_keys, (fragment.smiles, fragment.bond_list.tolist(), frag_bond.id))

    @staticmethod
    def get_tree_keys(fragment_mol, frag_bond_list: FragBondList, start_bond_pos):
        traversal_order = [[frag_bond_list[start_bond_pos].token]]
        visited = set()
        FragmentTree.dfs(fragment_mol, visited, traversal_order, prev_atom_indices=[frag_bond_list[start_bond_pos].atom_idx], frag_bond_list=FragBondList([frag_bond for frag_bond in frag_bond_list if frag_bond.id != start_bond_pos]))
        traversal_order = [sorted(v, key=lambda x: bond_priority.get(x, x)) for v in traversal_order]
        tree_keys = [','.join(map(str, v)) for v in traversal_order]
        return tree_keys

    @staticmethod
    def dfs(mol, visited, traversal_order, prev_atom_indices, frag_bond_list):
        next_atom_indices = []
        current_symbols = []
        current_bonds = []

        for atom_index in prev_atom_indices:
            if atom_index in visited:
                continue
            atom = mol.GetAtomWithIdx(atom_index)
            visited.add(atom_index)
            current_symbols.append(atom.GetAtomicNum())
            for frag_bond in frag_bond_list:
                if frag_bond.atom_idx == atom_index:
                    current_bonds.append(frag_bond.token)

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
            FragmentTree.dfs(mol, visited, traversal_order, next_atom_indices, frag_bond_list)

    # Example function to build the tree from a list of fragments
    def add_fragment_list(self, fragment_list: list[Fragment]):
        # Root node for the tree
        # root = FragmentNode('', 'Root')
        for fragment in tqdm(fragment_list, mininterval=0.5, desc='Building Fragment Tree'):
            self.add_fragment(fragment)

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


def split_fragment_info(ori_fragment: Fragment, cut_bond_indices):
    # Extract the SMILES string from frag_info
    smiles = ori_fragment.smiles

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
    bond_tokens_dict = {}
    for idx1, idx2 in cut_bond_indices:
        if new_mol.GetBondBetweenAtoms(idx1, idx2) is not None:
            atom1 = new_mol.GetAtomWithIdx(idx1)
            atom2 = new_mol.GetAtomWithIdx(idx2)
            bond = new_mol.GetBondBetweenAtoms(idx1, idx2)
            new_mol = add_Hs(new_mol, atom1, atom2, bond)  # Add hydrogens to adjust valence
            new_mol.RemoveBond(atom1.GetIdx(), atom2.GetIdx())
            if idx1 < idx2:
                bond_tokens_dict[(idx1, idx2)] = chem_bond_to_token(bond.GetBondType())
            else:
                bond_tokens_dict[(idx2, idx1)] = chem_bond_to_token(bond.GetBondType())
        else:
            raise ValueError(f"No bond found between atom indices {idx1} and {idx2}.")
    
    # Generate new fragment information for each resulting fragment
    new_mol = new_mol.GetMol()
    new_mol = sanitize(new_mol, kekulize = False)
    new_smiles = Chem.MolToSmiles(new_mol)
    fragment_mols = [Chem.MolFromSmiles(fragment) for fragment in new_smiles.split('.')]
    fragment_mols = [sanitize(fragment, kekulize = False) for fragment in fragment_mols]

    new_fragments = []
    bond_poses_dict = defaultdict(lambda: defaultdict(list))
    for frag_idx, frag_mol in enumerate(fragment_mols):
        
        atom_dict = {} # original atom index -> local atom index
        for i, atom in enumerate(frag_mol.GetAtoms()):
            amap = atom.GetAtomMapNum()
            atom_dict[amap] = i
            
        for atom in frag_mol.GetAtoms():
            frag_mol.GetAtomWithIdx(atom.GetIdx()).SetAtomMapNum(0)

        frag_smi = Chem.MolToSmiles(frag_mol)
        atom_order = list(map(int, frag_mol.GetProp('_smilesAtomOutputOrder')[1:-2].split(",")))

        for key in atom_dict: # original atom index -> new local atom index
            atom_dict[key] = atom_order.index(atom_dict[key])
        
        frag_bond_list = []
        for frag_bond in ori_fragment.bond_list:
            if frag_bond.atom_idx in atom_dict:
                frag_bond_list.append(FragmentBond(atom_dict[frag_bond.atom_idx], frag_bond.token))
        for (idx1, idx2) in cut_bond_indices:
            if idx1 in atom_dict:
                idx = idx1
            elif idx2 in atom_dict:
                idx = idx2
            else:
                continue
            
            bond_token = bond_tokens_dict.get((idx1, idx2), bond_tokens_dict.get((idx2, idx1)))
            frag_bond_list.append(FragmentBond(atom_dict[idx], bond_token))

        atom_map = [ori_fragment.atom_map[k] for k, v in sorted(atom_dict.items(), key=lambda x: x[1])]
        new_fragment = Fragment(frag_smi, FragBondList(frag_bond_list), atom_map)
        for frag_bond in new_fragment.bond_list:
            bond_poses_dict[frag_idx][(frag_bond.atom_idx, frag_bond.token)].append(frag_bond.id)
        new_fragments.append(new_fragment)
    
    fragment_group = FragmentGroup(new_fragments)
    bond_pair = {}
    for (bond_idx1, bond_idx2) in cut_bond_indices:
        for frag_idx, new_fragment in enumerate(new_fragments):
            if ori_fragment.atom_map[bond_idx1] in new_fragment.atom_map:
                bond_i1 = new_fragment.atom_map.index(ori_fragment.atom_map[bond_idx1])
                frag_idx1 = frag_idx
            if ori_fragment.atom_map[bond_idx2] in new_fragment.atom_map:
                bond_i2 = new_fragment.atom_map.index(ori_fragment.atom_map[bond_idx2])
                frag_idx2 = frag_idx
        bond = mol.GetBondBetweenAtoms(bond_idx1, bond_idx2)
        bond_token = chem_bond_to_token(bond.GetBondType())

        bond_pair[(len(bond_pair), bond_token)] = \
            ((frag_idx1, bond_i1), 
             (frag_idx2, bond_i2))
            # ((frag_idx1, bond_poses_dict[frag_idx1][(bond_i1, bond_token)].pop(0)), 
            #  (frag_idx2, bond_poses_dict[frag_idx2][(bond_i2, bond_token)].pop(0)))
    
    fragment_group.set_bond_pair(bond_pair)

    return fragment_group