import dill
from collections import Counter, defaultdict
import bidict
import torch

from .utils import *
from .fragment_bond import FragmentBond, FragBondList
from .fragment import Fragment
from .fragment_group import FragmentGroup
from .fragmentizer import Fragmentizer
from .fragment_tree import FragmentTree


class Vocab:
    def __init__(self, monoatomic_tokens_path, fragment_counter_path, threshold):
        # load monoatomic tokens and fragment counter
        with open(monoatomic_tokens_path, 'r') as f:
            monoatomic_fragments = [Fragment.parse_fragment_string(line.strip()) for line in f]
        if fragment_counter_path.endswith('.pkl'):
            fragment_counter = dill.load(open(fragment_counter_path, 'rb'))
        else:
            with open(fragment_counter_path, 'r') as f:
                fragment_counter = dict([(eval(line.strip().split('\t')[0]), int(line.strip().split('\t')[1]))
                        for line in f])
                fragment_counter = Counter(fragment_counter)

        # get the vocab
        vocab_dict = {tuple(frag): frag for frag in monoatomic_fragments if ':' not in frag.bond_list.tokens}
        # self.vocab = set([tuple(frag) for frag in monoatomic_fragments if ':' not in frag.bond_list.tokens])
        for frag_strings, cnt in fragment_counter.items():
            if cnt >= threshold:
                fragment = Fragment.parse_fragment_string(frag_strings)
                if fragment.mol is not None:
                    # self.vocab.add(tuple(fragment))
                    vocab_dict[tuple(fragment)] = fragment
                # else:
                #     print(f"Invalid fragment: {frag_strings}")
        # self.vocab.update([tuple(Fragment.parse_fragment_string(frag_strings)) for frag_strings, cnt in fragment_counter.items() if cnt >= threshold])

        self.vocab = bidict({v:i for i, v in enumerate(vocab_dict.keys())})

        self.tree = FragmentTree()
        self.tree.add_fragment_list(vocab_dict.values())
        
        self.fragmentizer = Fragmentizer()

    def __len__(self):
        return len(self.vocab)
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.vocab.inv[key]
        elif isinstance(key, tuple):
            return self.vocab[key]

    def assign_vocab(self, mol):
        fragment_group = self.fragmentizer.split_molecule(mol)

        frag_to_idx = {}
        for i, fragment in enumerate(fragment_group):
            if tuple(fragment) in frag_to_idx:
                continue
            idx = self.vocab.get(tuple(fragment), -1)
            if idx == -1:
                ring_info = fragment.mol.GetRingInfo()
                # Nonvalid fragment if it contains a ring
                try:
                    num_rings = ring_info.NumRings()
                except Exception as e:
                    raise ValueError(e)
                    fragment.mol = Chem.MolFromSmiles(fragment.smiles)
                    ring_info = fragment.mol.GetRingInfo()
                    num_rings = ring_info.NumRings()
                if num_rings > 0:
                    raise ValueError("Error: Ring not in vocabulary.")

            frag_to_idx[tuple(fragment)] = idx
        
        atom_scores = calculate_advanced_scores(mol)
        max_atom_score_tuple = max(atom_scores, key=lambda x: x[2])
        max_atom_score_idx = max_atom_score_tuple[0]
        # max_atom_score = max_atom_score_tuple[2]

        ori_frag_idx, ori_atom_idx = fragment_group.match_atom_map(max_atom_score_idx)

        vocab_list = []
        visited = set()
        current_frag_idx = -1

        def dfs(parent_info, start_bond_info):
            nonlocal current_frag_idx
            frag_idx, s_bond_idx = start_bond_info
            fragment: Fragment = fragment_group[frag_idx]
            atom_mapping = fragment.atom_map

            vocab_idx = frag_to_idx[tuple(fragment)]
            if vocab_idx == -1:
                current_frag_idx += 1
                order_frag_idx = current_frag_idx
                visited.update(atom_mapping)
                if s_bond_idx == -1:
                    start_atom_idx = atom_mapping.index(max_atom_score_idx)
                else:
                    start_atom_idx = None
                result =  self.tree.search(fragment, s_bond_idx, start_atom_idx=start_atom_idx)
                if result is None:
                    return None
                root_next, sub_vocab_list, local_atom_map = result
                current_bond_pos = root_next[2][1]
                merge_bond_poses = []
                for sub_vocab_idx, sub_vocab in enumerate(sub_vocab_list):
                    tmp_frag = sub_vocab['frag']
                    if tuple(tmp_frag) not in frag_to_idx:
                        frag_to_idx[tuple(tmp_frag)] = self.vocab.get(tuple(tmp_frag), -1)
                    sub_vocab['idx'] = frag_to_idx[tuple(tmp_frag)]

                    for i, next_v in enumerate(sub_vocab['next']):
                        sub_vocab['next'][i] = (next_v[0], next_v[1], (next_v[2][0] + current_frag_idx, next_v[2][1]))

                    for frag_bond in tmp_frag.bond_list:
                        merge_bond_poses.append((sub_vocab_idx, frag_bond.id))

                    vocab_list.append(sub_vocab)
                if root_next[2][1] != -1:
                    merge_bond_poses.remove((root_next[2][0], root_next[2][1]))
                for sub_vocab_idx, sub_vocab in enumerate(sub_vocab_list):
                        for i, next_v in enumerate(sub_vocab['next']):
                            merge_bond_poses.remove((sub_vocab_idx, next_v[0]))
                            merge_bond_poses.remove((next_v[2][0] - current_frag_idx, next_v[2][1]))
                    
                next_atom_infoes = []
                tmp = defaultdict(list)
                for i, (sub_vocab_idx, bond_pos) in enumerate(merge_bond_poses):
                    atom_idx = local_atom_map.inv[(sub_vocab_idx, sub_vocab_list[sub_vocab_idx]['frag'].bond_list[bond_pos].atom_idx)]
                    bond_token = sub_vocab_list[sub_vocab_idx]['frag'].bond_list[bond_pos].token

                    vocab_idx = sub_vocab_idx+current_frag_idx
                    tmp[(atom_idx, bond_token)].append((vocab_idx, bond_pos))
                
                for frag_bond in fragment.bond_list:
                    if frag_bond.id == s_bond_idx:
                        continue
                    vocab_idx, bond_pos = tmp[(fragment.atom_map[frag_bond.atom_idx], frag_bond.token)].pop(0)
                    next_frag_idx, next_bond_pos = fragment_group.get_neighbor(frag_idx, frag_bond.id)

                    next_atom_infoes.append(((vocab_idx, bond_pos), bond_token, (next_frag_idx, next_bond_pos)))
                
                current_frag_idx += len(sub_vocab_list) - 1

                for (vocab_idx, bond_pos), bond_token, (next_frag_idx, next_bond_pos) in next_atom_infoes:
                    next_frag_idx, next_bond_pos = dfs(parent_info=(vocab_idx, bond_pos), start_bond_info=(next_frag_idx, next_bond_pos))
                    vocab_list[vocab_idx]['next'].append((bond_pos, bond_token, (next_frag_idx, next_bond_pos)))
                    
            else:
                vocab_list.append({'frag': tuple(fragment), 'idx': vocab_idx, 'next': []})
                current_frag_idx += 1
                order_frag_idx = current_frag_idx
                current_bond_pos = s_bond_idx
                visited.update(atom_mapping)

                frag_neighbors = fragment_group.get_neighbors(frag_idx)
                frag_neighbors = {from_bond_pos: (to_frag_idx, to_bond_pos) for from_bond_pos, (to_frag_idx, to_bond_pos) in frag_neighbors.items() if from_bond_pos != s_bond_idx}

                for cur_bond_pos, (next_frag_idx, next_bond_pos) in frag_neighbors.items():
                    bond_token = fragment_group.bond_token_between(frag_idx, cur_bond_pos, next_frag_idx, next_bond_pos)
                    next_frag_idx, next_bond_pos = dfs(parent_info=(order_frag_idx, cur_bond_pos), start_bond_info=(next_frag_idx, next_bond_pos))
                    vocab_list[order_frag_idx]['next'].append((cur_bond_pos, bond_token, (next_frag_idx, next_bond_pos)))

                
            return order_frag_idx, current_bond_pos
            
        result = dfs(parent_info=(-1,-1), start_bond_info=(ori_frag_idx, -1))

        if result is None:
            return None
        
        return vocab_list
    

    def tensorize(self, mol, max_seq_len = 100):
        vocab_tree = self.assign_vocab(mol)
        # print('\n'.join([str(i) + ': ' + str(vt) for i, vt in enumerate(vocab_tree)]))
        if vocab_tree is None:
            return None
        
        vocab_tensor = torch.empty(max_seq_len, dtype=torch.int32)
        order_tensor = torch.empty(max_seq_len, 3, dtype=torch.int32) # (parent_idx, parent_bond_pos, bond_pos)
        mask_tensor =  torch.zeros(max_seq_len, dtype=torch.bool)  # 初期値は False
        mask_tensor[:len(vocab_tree)] = True

        parent_data = {}
        parent_data[0] = (-1, -1, -1)
        for i, vocab in enumerate(vocab_tree):
            for next_vocab in vocab['next']:
                parent_data[next_vocab[2][0]] = (i, next_vocab[0], next_vocab[2][1]) # (parent_idx, parent_bond_pos, bond_pos)

        for i, vocab in enumerate(vocab_tree):
            vocab_tensor[i] = vocab['idx']
            order_tensor[i] = torch.tensor(parent_data[i], dtype=torch.int32)

        return vocab_tensor, order_tensor, mask_tensor

    def detensorize(self, vocab_tensor, order_tensor, mask_tensor):
        # 空の vocab_tree を作成
        vocab_tree = []

        # mask_tensor で有効なインデックスのみ処理
        valid_indices = mask_tensor.nonzero(as_tuple=True)[0]

        # vocab_tree を再構築
        for idx in valid_indices:
            vocab_idx = vocab_tensor[idx].item()
            parent_idx, parent_bond_pos, bond_pos = order_tensor[idx].tolist()
            
            # ノードのデータ構造
            frag_info = self.vocab.inv[vocab_idx]
            frag = Fragment.from_tuple(frag_info)
            node = {
                'frag': frag, 
                'idx': vocab_idx,
                'next': []
            }
            vocab_tree.append(node)

            # 親ノードに 'next' 情報を追加
            if parent_idx >= 0:
                vocab_tree[parent_idx]['next'].append((parent_bond_pos, frag.bond_list[bond_pos].token, (idx.item(), bond_pos)))

        # 再構築された vocab_tree を返す
        mol = self.vocab_tree_to_mol(vocab_tree)
        return mol

    def vocab_tree_to_mol(self, vocab_tree):
        merge_bond_poses = []
        fragments = []
        for frag_id1, vocab in enumerate(vocab_tree):
            for next_info in vocab['next']:
                bond_pos1, bond_type, (frag_id2, bond_pos2) = next_info
                merge_bond_poses.append(((frag_id1, bond_pos1), bond_type, (frag_id2, bond_pos2)))
            fragments.append(vocab['frag'])
        
        merged_frag =  merge_fragment_info(fragments, merge_bond_poses)

        # 分子を正規化して返す
        mol = Chem.MolFromSmiles(merged_frag.smiles)
        return mol

def calculate_advanced_scores(smiles_or_mol: str, sort=False):
    """
    Calculate advanced scores for each atom in a molecule based on:
    - Symbol score (atomic type)
    - Join type score (bond type)
    - Distance from molecule center
    """
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    elif isinstance(smiles_or_mol, Chem.Mol):
        mol = smiles_or_mol
    else:
        raise ValueError("Invalid input type. Expected SMILES string or RDKit Mol object.")
    
    if mol is None:
        raise ValueError(f"Invalid mol")

    total_atoms = mol.GetNumAtoms()
    distance_matrix = Chem.rdmolops.GetDistanceMatrix(mol)

    # Updated symbol_scores with Carbon having the highest score
    symbol_scores = {
        "C": 2.0, "N": 1.5, "O": 1.4, "S": 1.3, "P": 1.2, 
        "F": 1.1, "Cl": 1.0, "Br": 0.9, "I": 0.8, 
        "Si": 0.7, "B": 0.6, "Li": 0.5, "Na": 0.4, 
        "K": 0.3, "Mg": 0.2, "Ca": 0.1
    }

    bond_scores = {Chem.BondType.SINGLE: 1.0, Chem.BondType.DOUBLE: 1.5, Chem.BondType.TRIPLE: 2.0, Chem.BondType.AROMATIC: 2.5}

    scores = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        if symbol == "H":  # Skip hydrogen
            continue
        max_distance = distance_matrix[atom_idx].max()  # Average distance
        symbol_score = symbol_scores.get(symbol, 0.1)  # Default to minimum score for unknown atoms

        # Calculate join type score
        join_type_score = sum(
            bond_scores.get(bond.GetBondType(), 0) for bond in atom.GetBonds()
        )

        # Calculate the final score
        atom_score = sum(
            ((max_distance - dist) / max_distance / join_type_score * symbol_score) / total_atoms
            for dist in distance_matrix[atom_idx]
        )
        scores.append((atom_idx, symbol, atom_score))
    
    if sort:
        scores = sorted(scores, key=lambda x: x[2], reverse=True)
    return scores


def merge_fragment_info(fragments: list[Fragment], merge_bond_poses, atom_id_list = None):
    """
    Merge multiple molecular fragments into a single molecule by combining and bonding them.
    
    Args:
        frag_infoes (list of tuples): Each tuple contains fragment information. 
            The first element is the SMILES string of the fragment. Subsequent elements are:
            (smiles, atom_idx1, bond_type1, atom_idx2, bond_type2, ...)
                - The atom indices involved in bonds
                - The bond types (e.g., '-', '=', etc.)
        merge_bond_poses (list of tuples): Specifies the bonds to be created between fragments.
            Each tuple contains:
            ((frag_idx1, bond_pos1), bond_type, (frag_idx2, bond_pos2))
                - Position of the first fragment and its bond index
                - Bond type (e.g., '-', '=', etc.)
                - Position of the second fragment and its bond index
        atom_id_list (list of lists, optional): Maps fragment atom indices to global atom IDs.
            If not provided, default indices [0, 1, ..., N] are used for each fragment.

    Returns:
        tuple: A tuple containing:
            - final_frag_info (tuple): The SMILES string of the combined molecule and bond information.
            - final_atom_map (bidict): Maps global atom indices in the combined molecule to fragment indices and atom IDs.
    """

    # Convert SMILES to RDKit molecules for each fragment
    mols = [Chem.MolFromSmiles(frag.smiles) for frag in fragments]

    # If no atom_id_list is provided, use default atom indices for each fragment
    if atom_id_list is None:
        atom_id_list = [list(range(mol.GetNumAtoms())) for mol in mols]

    # Initialize atom mapping and remaining bond positions
    atom_map = bidict()
    remaining_bond_poses = []
    offset = 1

    # Combine molecules and assign atom map numbers
    for i, mol in enumerate(mols):
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom.SetAtomMapNum(atom_idx + offset)  # Assign unique atom map numbers
            atom_map[atom_idx + offset] = (i, atom_idx)
            
        if i == 0:
            combined_mol = copy.deepcopy(mol)  # Start with the first fragment
        else:
            combined_mol = Chem.CombineMols(combined_mol, mol)  # Add subsequent fragments

        # Track remaining bonds in the fragment
        for frag_bond in fragments[i].bond_list:
            remaining_bond_poses.append((i, frag_bond.id))
        offset += mol.GetNumAtoms()  # Update offset for the next fragment

    # Convert the combined molecule to an editable RWMol
    combined_rwmol = Chem.RWMol(combined_mol)

    # Add specified bonds between fragments
    for i, (joint_pos1, bond_token, joint_pos2) in enumerate(merge_bond_poses):
        frag_idx1, bond_pos1 = joint_pos1
        map_number1 = atom_map.inv[(frag_idx1, fragments[frag_idx1].bond_list[bond_pos1].atom_idx)]

        frag_idx2, bond_pos2 = joint_pos2
        map_number2 = atom_map.inv[(frag_idx2, fragments[frag_idx2].bond_list[bond_pos2].atom_idx)]

        # Find atom indices by map number
        atom_idx1 = next(atom.GetIdx() for atom in combined_rwmol.GetAtoms() if atom.GetAtomMapNum() == map_number1)
        atom_idx2 = next(atom.GetIdx() for atom in combined_rwmol.GetAtoms() if atom.GetAtomMapNum() == map_number2)

        atom1 = combined_rwmol.GetAtomWithIdx(atom_idx1)
        atom2 = combined_rwmol.GetAtomWithIdx(atom_idx2)
        bond_type = token_to_chem_bond(bond_token)  # Convert bond type to RDKit format
        combined_rwmol.AddBond(atom_idx1, atom_idx2, bond_type)  # Add bond
        combined_rwmol = remove_Hs(combined_rwmol, atom1, atom2, bond_type)  # Remove hydrogens
        remaining_bond_poses.remove((frag_idx1, bond_pos1))
        remaining_bond_poses.remove((frag_idx2, bond_pos2))

    # Generate the final combined molecule and SMILES
    combined_mol = combined_rwmol.GetMol()
    atom_map2 = bidict() # combined_order -> (frag_idx, pre_atom_idx)
    for i, atom in enumerate(combined_mol.GetAtoms()):
        atom_map2[i] = atom_map[atom.GetAtomMapNum()]
    for atom in combined_mol.GetAtoms():
        atom.SetAtomMapNum(0)
    smiles = Chem.MolToSmiles(combined_mol, isomericSmiles=True)

    # Extract atom order from SMILES
    atom_order = list(map(int, combined_mol.GetProp('_smilesAtomOutputOrder')[1:-2].split(",")))

    new_atom_maps = bidict({i: atom_map2[order] for i, order in enumerate(atom_order)})

    # Collect remaining bond information
    # (frag_idx, atom_idx) -> atom_map_num
    bond_list = []
    for frag_idx, bond_pos in remaining_bond_poses:
        frag_bond = fragments[frag_idx].bond_list[bond_pos]
        bond_list.append((atom_order.index(atom_map2.inv[(frag_idx, atom_id_list[frag_idx][frag_bond.atom_idx])]), frag_bond.token))
    bond_list = FragBondList(bond_list)

    # Create the new fragment information
    new_fragment = Fragment(smiles, bond_list, list(range(len(new_atom_maps))))

    return new_fragment