from collections import defaultdict
import copy
from rdkit import Chem

from .utils import *
from .fragment import Fragment, FragBondList, FragmentBond

class FragmentGroup:
    def __init__(self, fragments:list[Fragment]):
        self.fragments = []
        for id, fragment in enumerate(fragments):
            if not isinstance(fragment, Fragment):
                raise ValueError("Only Fragment instances can be added.")
            fragment.id = id
            self.fragments.append(fragment)

        self._neighbors = {fragment.id: {} for fragment in self.fragments}
        self._bond_between = {}

    def __str__(self):
        strings = [f'{fragment.id}:'+ str(tuple(fragment)) + ' --> ' + str(self.get_neighbors(fragment.id)) for fragment in self.fragments]
        return '\n'.join(strings)
    
    def __repr__(self):
        return self.__str__()

        
    def __len__(self):
        return len(self.fragments)
    
    def __iter__(self):
        return iter(self.fragments)
    
    def __getitem__(self, key):
        return self.fragments[key]
    
    def get_bond_between(self, frag_idx1, bond_pos1, frag_idx2, bond_pos2):
        bond_token1 = self.fragments[frag_idx1].bond_list[bond_pos1].token
        bond_token2 = self.fragments[frag_idx2].bond_list[bond_pos2].token
        if bond_token1 != bond_token2:
            raise ValueError(f"Bond tokens do not match: {bond_pos1} of {self.fragments[frag_idx1]} != {bond_pos2} of {self.fragments[frag_idx2]}")
        return copy.deepcopy(self.fragments[frag_idx1].bond_list[bond_pos1])

    def set_bond_pair(self, bond_pair):
        """
        bond_pair: dict of bond pairs {(sep_idx, bond_token): ((frag_idx1, atom_idx1), (frag_idx2, atom_idx2)), ...}
        """
        bond_positions = []
        start_bond_pos_dict = defaultdict(int)
        for (sep_idx, bond_token), ((frag_idx1, atom_idx1), (frag_idx2, atom_idx2)) in bond_pair.items():
            bond_pos1 = self.fragments[frag_idx1].get_bond_pos(atom_idx1, bond_token, start_pos=start_bond_pos_dict[(frag_idx1, atom_idx1, bond_token)])
            bond_pos2 = self.fragments[frag_idx2].get_bond_pos(atom_idx2, bond_token, start_pos=start_bond_pos_dict[(frag_idx2, atom_idx2, bond_token)])
            bond_positions.append(((frag_idx1, bond_pos1), bond_token, (frag_idx2, bond_pos2)))
            start_bond_pos_dict[(frag_idx1, atom_idx1, bond_token)] += 1
            start_bond_pos_dict[(frag_idx2, atom_idx2, bond_token)] += 1

        self.set_bond_positions(bond_positions)
    
    def set_bond_positions(self, bond_positions):
        """
        bond_positions: list of bond positions [((frag_idx1, bond_pos1), bond_token, (frag_idx2, bond_pos2)), ...]
        """
        self._neighbors = {fragment.id: {} for fragment in self.fragments}
        for (frag_idx1, bond_pos1), bond_token, (frag_idx2, bond_pos2) in bond_positions:
            if frag_idx1 >= len(self.fragments):
                raise ValueError(f"Fragment index {frag_idx1} is out of range.")
            if frag_idx2 >= len(self.fragments):
                raise ValueError(f"Fragment index {frag_idx2} is out of range.")
            if bond_pos1 >= len(self.fragments[frag_idx1].bond_list):
                raise ValueError(f"Frag{frag_idx1} Bond position {bond_pos1} is out of range.")
            if bond_pos2 >= len(self.fragments[frag_idx2].bond_list):
                raise ValueError(f"Frag{frag_idx2} Bond position {bond_pos2} is out of range.")

            if frag_idx1 not in self._neighbors:
                self._neighbors[frag_idx1] = {}
            self._neighbors[frag_idx1][bond_pos1] = (frag_idx2, bond_pos2)

            if frag_idx2 not in self._neighbors:
                self._neighbors[frag_idx2] = {}
            self._neighbors[frag_idx2][bond_pos2] = (frag_idx1, bond_pos1)

            self._bond_between[((frag_idx1, bond_pos1), (frag_idx2, bond_pos2))] = bond_token
            self._bond_between[((frag_idx2, bond_pos2), (frag_idx1, bond_pos1))] = bond_token
        
    def get_neighbors(self, frag_idx) -> dict:
        return {s_bond_pos: (e_frag_idx, e_bond_pos) for s_bond_pos, (e_frag_idx, e_bond_pos) in self._neighbors[frag_idx].items()}

    
    def get_neighbor(self, frag_idx, bond_pos):
        neighbor_frag_idx, neighbor_bond_pos = self._neighbors[frag_idx][bond_pos]
        return neighbor_frag_idx, neighbor_bond_pos
    
    def bond_token_between(self, frag_idx1, bond_pos1, frag_idx2, bond_pos2):
        return self._bond_between[((frag_idx1, bond_pos1), (frag_idx2, bond_pos2))]

    def match_atom_map(self, atom_map_num: int):
        for fragment in self.fragments:
            if atom_map_num in fragment.atom_map:
                idx = fragment.atom_map.index(atom_map_num)
                return fragment.id, idx
        raise ValueError(f"Atom map number {atom_map_num} not found.")


    


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