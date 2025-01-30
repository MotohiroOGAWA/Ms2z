from __future__ import annotations

from rdkit import Chem
import copy
from collections import defaultdict, Counter
import dill
import os
import re
import torch

from .utils import *
from .fragment_bond import *



class Fragment:
    # Map bond types to their respective annotations
    bond_annotations = {'-': '[Nh]', '=': '[Og]', '#': '[Ts]', ':': '[Lv]'}

    def __init__(self, smiles_or_mol, frag_bond_list:FragBondList|list=[], atom_map=None):
        if isinstance(smiles_or_mol, str):
            smiles = smiles_or_mol
        elif isinstance(smiles_or_mol, Chem.Mol):
            smiles = Chem.MolToSmiles(smiles_or_mol)
        else:
            raise ValueError("The input must be either a SMILES string or an RDKit molecule object.")
        
        if isinstance(frag_bond_list, list):
            frag_bond_list = FragBondList(frag_bond_list)
        elif not isinstance(frag_bond_list, FragBondList):
            raise ValueError("The bond list must be an instance of FragBondList.")

        mol, smiles, frag_bond_list, atom_map, \
            mol_with_alt, smiles_with_alt, atom_map_with_alt \
                = Fragment.normalize_frag_info(smiles, frag_bond_list, atom_map)

        self.mol = Chem.MolFromSmiles(smiles)
        self.smiles = smiles
        if isinstance(frag_bond_list, FragBondList):
            self.bond_list: FragBondList = frag_bond_list
        else:
            raise ValueError("The bond list must be an instance of FragBondList.")
        self.atom_map = atom_map
        self.id  = -1
        self.mol_with_alt = mol_with_alt
        self.smiles_with_alt = smiles_with_alt
        self.atom_map_with_alt = atom_map_with_alt
        self.smarts_mol = None
        
    def __str__(self):
        output = [self.smiles] + [item for atom_idx_bond_type in self.bond_list for item in atom_idx_bond_type]
        return str(tuple(output))

    def __repr__(self):
        return str(self.__str__())
    
    def __iter__(self):
        output = [self.smiles] + [item for atom_idx_bond_type in self.bond_list for item in atom_idx_bond_type]
        return iter(tuple(output))
    
    def __eq__(self, other: Fragment):
        return tuple(self) == tuple(other)

    def get_smiles_with_map(self):
        mol = Chem.MolFromSmiles(self.smiles)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(self.atom_map[atom.GetIdx()])
        return Chem.MolToSmiles(mol)
    
    def get_smiles_with_idx(self):
        mol = Chem.MolFromSmiles(self.smiles)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        return Chem.MolToSmiles(mol)
    
    def get_smiles_with_alt_map(self):
        mol = Chem.MolFromSmiles(self.smiles_with_alt)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(self.atom_map_with_alt[atom.GetIdx()])
        return Chem.MolToSmiles(mol)
    
    def get_smiles_with_alt_idx(self):
        mol = Chem.MolFromSmiles(self.smiles_with_alt)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        return Chem.MolToSmiles(mol)

    def get_bond_pos(self, atom_idx, bond_token, start_pos=0):
        match_bond_ids = self.bond_list.get_bond_ids(atom_idx, bond_token)
        return match_bond_ids[start_pos]
    
    def get_bond_poses(self, atom_idx, bond_token):
        return self.bond_list.get_bond_ids(atom_idx, bond_token)

    def get_bond_poses_batch(self, atom_bond_list):
        """
        Retrieve bond positions for a list of atom indices and bond tokens.

        Args:
            atom_bond_list (list[tuple]): List of tuples containing atom index and bond token. 
            [(atom_idx, bond_token), ...]

        Returns:
            list[int]: List of bond positions corresponding to the input atom-bond pairs.
        """
        bond_start_indices = defaultdict(int)  # Tracks the start index for each atom-bond pair
        bond_positions = []

        for atom_idx, bond_token in atom_bond_list:
            bond_pos = self.get_bond_pos(atom_idx, bond_token, start_pos=bond_start_indices[(atom_idx, bond_token)])
            bond_start_indices[(atom_idx, bond_token)] += 1
            bond_positions.append(bond_pos)

        return bond_positions

    def get_frag_bond_tensor(self):
        atom_num = self.mol.GetNumAtoms()
        frag_bond_tensor = torch.zeros(atom_num, 3, dtype=torch.int32)
        for bond in self.bond_list:
            frag_bond_tensor[bond.atom_idx, bond.num - 1] += 1
        return frag_bond_tensor

    def GetSubstructMatches(self, query: Fragment):
        matches = self.mol_with_alt.GetSubstructMatches(query.smarts) # qry_alt_idx: tgt_alt_idx
        matches = list(set(matches))

        matches2 = []
        for match in matches:
            tmp = [m for i, m in enumerate(match) if query.atom_map_with_alt[i] >= 0]
            adjacent_pairs = search_atom_indices_adjacent_to_group(self.mol_with_alt, tmp)
            if len(adjacent_pairs) == len(query.bond_list):
                matches2.append(match)


        qry_idx_to_tgt_alt_idx_list = []
        for match in matches2:
            qry_idx_to_tgt_alt_idx = []
            for qry_idx in query.atom_map:
                qry_idx_to_tgt_alt_idx.append(match[query.atom_map_with_alt.index(qry_idx)])
            qry_idx_to_tgt_alt_idx_list.append(qry_idx_to_tgt_alt_idx)
        
        qry_idx_to_tgt_idx_list = []
        for qry_idx_to_tgt_alt_idx in qry_idx_to_tgt_alt_idx_list:
            match = []
            for tgt_alt_idx in qry_idx_to_tgt_alt_idx:
                match.append(self.atom_map_with_alt[tgt_alt_idx])
            qry_idx_to_tgt_idx_list.append(tuple(match))

        return qry_idx_to_tgt_idx_list

    def to_tuple(self):
        r = []
        r.append(self.smiles)
        for frag_bond in self.bond_list:
            r.append(frag_bond.atom_idx)
            r.append(frag_bond.token)
        return tuple(r)

    @staticmethod
    def from_tuple(fragment_tuple, atom_map=None):
        smiles = fragment_tuple[0]
        bond_infoes = [(int(atom_idx), bond_token) for atom_idx, bond_token in zip(fragment_tuple[1::2], fragment_tuple[2::2])]
        frag_bond_list = FragBondList(bond_infoes)
        return Fragment(smiles, frag_bond_list, atom_map)

    @staticmethod
    def normalize_frag_info(smiles, bond_list:FragBondList, atom_map = None):
        mol = Chem.MolFromSmiles(smiles)
        mol_with_alt, smiles_with_alt, atom_map_with_alt = Fragment._add_bond_annotations(mol, bond_list, atom_map)
        normalized_mol, normalized_smiles, normalized_atom_map, normalized_bond_list = Fragment._remove_bond_annotations(mol_with_alt, atom_map_with_alt)
        return normalized_mol, normalized_smiles, normalized_bond_list, normalized_atom_map, mol_with_alt, smiles_with_alt, atom_map_with_alt

    @staticmethod
    def _add_bond_annotations(mol, bond_list: FragBondList, atom_indices = None):
        mol = copy.deepcopy(mol)
        if atom_indices is None:
            atom_indices = list(range(mol.GetNumAtoms()))
        
        map_num_to_ori_atom_idx = {}
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(i+1)
            map_num_to_ori_atom_idx[i+1] = atom_indices[i]

        rwmol = Chem.RWMol(mol)
        atom_idx_offfset = mol.GetNumAtoms() + 1
        for frag_bond in bond_list:
            atom_idx = [atom.GetIdx() for atom in rwmol.GetAtoms() if atom.GetAtomMapNum() == frag_bond.atom_idx + 1][0]
            atom = rwmol.GetAtomWithIdx(atom_idx)
            new_atom = Chem.Atom(alt_atom[frag_bond.token])
            new_atom.SetAtomMapNum(atom_idx_offfset + frag_bond.id)

            rwmol = add_atom(rwmol, atom, new_atom, bond_type=frag_bond.type)
            map_num_to_ori_atom_idx[atom_idx_offfset + frag_bond.id] = - (frag_bond.id + 1)
        
        new_mol = rwmol.GetMol()
        if new_mol is None:
            raise ValueError(f"The molecule appears to be None, possibly due to incorrect substitution {Chem.MolToSmiles(rwmol)} of alternative atoms.")
        atom_map = {}
        for i, atom in enumerate(new_mol.GetAtoms()):
            atom_map[i] = atom.GetAtomMapNum()
        for atom in new_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        
        new_mol, new_smiles, atom_order = get_atom_order(new_mol)
        new_atom_map = {map_num_to_ori_atom_idx[atom_map[idx]]: i for i, idx in enumerate(atom_order)}
        new_atom_map = [k for k, v in sorted(new_atom_map.items(), key=lambda item: item[1])]

        return new_mol, new_smiles, new_atom_map
    
    def add_alt_bond(self, frag_bond_list):
        new_mol = copy.deepcopy(self.mol)
        new_frag_bond_list = copy.deepcopy(self.bond_list)
        new_atom_map = copy.deepcopy(self.atom_map)
        for frag_bond in frag_bond_list:
            frag_bond = FragmentBond(frag_bond.atom_idx, frag_bond.bond_token)
            new_frag_bond_list.add(frag_bond)
        new_frag = Fragment(new_mol, new_frag_bond_list, new_atom_map)
        return new_frag


    @staticmethod
    def _remove_bond_annotations(mol, atom_indices=None):
        """
        Removes bond annotations from a molecule.

        Args:
            mol (Chem.Mol): RDKit molecule object with bond annotations.

        Returns:
            Chem.Mol: RDKit molecule object without bond annotations.
        """
        mol = copy.deepcopy(mol)
        if atom_indices is None:
            atom_indices = list(range(mol.GetNumAtoms()))

        atom_map_to_idx = {}
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(atom.GetIdx()+1)
            atom_map_to_idx[atom.GetAtomMapNum()] = atom_indices[i]
        
        bond_atom_indices = defaultdict(list)
        rwmol = Chem.RWMol(mol)
        bond_atom_idx_pairs = []
        for i, atom in enumerate(mol.GetAtoms()):
            for bond_type, alt_symbol in alt_atom.items():
                if atom.GetSymbol() == alt_symbol.GetSymbol(): # Check if the atom is an alternative atom
                    neighbor = atom.GetNeighbors()[0]
                    bond_atom_idx_pairs.append((neighbor.GetAtomMapNum(), atom.GetAtomMapNum()))
                    # bond_atom_idx_pairs.append((neighbor.GetIdx(), atom.GetIdx()))
                    bond_atom_indices[neighbor.GetAtomMapNum()].append(bond_type)
                    break
        
        bond_atom_idx_pairs = sorted(bond_atom_idx_pairs, key=lambda x: x[1], reverse=True) # Sort by alternative atom index
        for (atom_map_num, alt_atom_map_num) in bond_atom_idx_pairs:
            alt_atom_idx = [atom.GetIdx() for atom in rwmol.GetAtoms() if atom.GetAtomMapNum() == alt_atom_map_num][0]
            atom_idx = [atom.GetIdx() for atom in rwmol.GetAtoms() if atom.GetAtomMapNum() == atom_map_num][0]
        # for (atom_idx, alt_atom_idx) in bond_atom_idx_pairs:
            bond = rwmol.GetBondBetweenAtoms(alt_atom_idx, atom_idx)
            if bond:
                removed_atom = rwmol.GetAtomWithIdx(alt_atom_idx)
                neighbor_atom = rwmol.GetAtomWithIdx(atom_idx)
                rwmol = remove_atom(rwmol, removed_atom, neighbor_atom, bond.GetBondType())
                # rwmol = add_Hs(rwmol, rwmol.GetAtomWithIdx(atom_idx), a2=None, bond=bond)
                # rwmol = Chem.RWMol(Chem.RemoveHs(rwmol))

        new_mol = rwmol.GetMol()
        atom_map = {}
        map_num_to_pre_order = {}
        for i, atom in enumerate(new_mol.GetAtoms()):
            atom_map[i] = atom_map_to_idx[atom.GetAtomMapNum()]
            map_num_to_pre_order[atom.GetAtomMapNum()] = i
        for atom in new_mol.GetAtoms():
            atom.SetAtomMapNum(0)

        new_mol, smiles, atom_order = get_atom_order(new_mol)
        new_atom_map = {atom_map[idx]: i for i, idx in enumerate(atom_order)}
        new_atom_map = [k for k, v in sorted(new_atom_map.items(), key=lambda item: item[1])]

        bond_list = FragBondList()
        for atom_map_num, bond_token_list in bond_atom_indices.items():
            for bond_token in bond_token_list:
                fragment_bond = FragmentBond(atom_order.index(map_num_to_pre_order[atom_map_num]), bond_token)
                bond_list.add(fragment_bond)

        return new_mol, smiles, new_atom_map, bond_list

    
    @staticmethod
    def split_to_monoatomic_fragment(mol, aromatic_bonds=True):
        fragments = []
        for atom in mol.GetAtoms():
            atom_symbol = get_atom_symbol(atom)  # Get the symbol of the atom (e.g., 'C', 'O')
            if atom_symbol == "H":  # Skip hydrogen
                continue
            
            # Find the first bond type for this atom
            bonds = atom.GetBonds()
            bond_tokens = []
            if bonds:
                for bond in bonds:  # Process all bonds for the atom
                    bond_type = bond.GetBondType()
                    bond_token = chem_bond_to_token(bond_type)
                    bond_tokens.append(bond_token)

                bond_tokens = sorted(bond_tokens, key=lambda x: bond_priority[x])
                
            bond_infoes = [(0, bond_token) for bond_token in bond_tokens]
            frag_bond_list = FragBondList(bond_infoes)
            if not aromatic_bonds and ':' in frag_bond_list.tokens:
                continue

            fragments.append(Fragment(get_atom_symbol(atom), frag_bond_list))

        return fragments
    
    @staticmethod
    def get_monoatomic_tokens(mol):
        monoatomic_fragments = Fragment.split_to_monoatomic_fragment(mol)
        monoatomic_tokens = list(set([str(fragment) for fragment in monoatomic_fragments if fragment.mol is not None]))
        return monoatomic_tokens
    
    @staticmethod
    def parse_fragment_string(fragment_string):
        """
        Parses a fragment string representation into a Fragment object.

        Args:
            fragment_string (str): The string representation of a fragment,
                                e.g., "('CC(C)C', 3, '-')".

        Returns:
            Fragment: A Fragment object with the parsed SMILES string and bond information.
        """
        fragment_info = tuple(eval(fragment_string))

        # Extract the SMILES string
        smiles = fragment_info[0]

        # Extract bond information as a list of tuples (integer, string)
        bond_infoes = [(int(atom_idx), bond_token) for atom_idx, bond_token in zip(fragment_info[1::2], fragment_info[2::2])]

        # Create the FragBondList object
        frag_bond_list = FragBondList(bond_infoes)

        # Create and return the Fragment object
        fragment_object = Fragment(smiles, frag_bond_list)

        return fragment_object
    
    @staticmethod
    def save_fragment_obj(fragment_list: list[Fragment], path, binary=True, text=False):
        base_name = os.path.splitext(path)[0]
        if binary:
            dill.dump(fragment_list, open(base_name + '.pkl', 'wb'))
        if text:
            with open(base_name + '.txt', 'w') as f:
                for fragment in fragment_list:
                    f.write(str(fragment) + '\n')

    @staticmethod
    def load_fragment_obj(path):
        if path.endswith('.pkl'):
            fragment_list = dill.load(open(path, 'rb'))
        else:
            with open(path, 'r') as f:
                fragment_list = [Fragment.parse_fragment_string(line.strip()) for line in f]
        return fragment_list

    @property
    def smarts(self):
        if self.smarts_mol is None:
            smarts_with_alt = Chem.MolToSmarts(self.mol_with_alt)
            for bond_token, alt_symbol in Fragment.bond_annotations.items():
                smarts_with_alt = smarts_with_alt.replace(alt_symbol, '[*]')
            self.smarts_mol = Chem.MolFromSmarts(smarts_with_alt)
        return self.smarts_mol



def search_atom_indices_adjacent_to_group(mol, atom_idx_group: list[int]):
    """
    Search for atom indices adjacent to a specified group of atomic indices in a molecule.

    Parameters:
        mol: RDKit molecule object
            The molecular structure in which the search is performed.
        atom_idx_group: list[int]
            A list of atomic indices representing the group for which adjacent atoms are to be identified.

    Returns:
        result: list[tuple[int, int]]
            A list of tuples where each tuple contains:
            - An atomic index from the specified group.
            - The atomic index of a neighboring atom that is not in the specified group.

    Description:
        This function iterates over the atoms in the specified `atom_idx_group` and identifies all neighboring
        atoms in the molecule `mol` that are not part of the given group. Each pair of a group atom and its
        adjacent non-group atom is returned as a tuple.

    Example:
        # Assuming `mol` is an RDKit molecule object
        atom_group = [0, 1, 2]
        adjacent_atoms = search_atom_indices_adjacent_to_group(mol, atom_group)
        print(adjacent_atoms)
        # Output might be: [(0, 3), (1, 4), (2, 5)]
    """
    result = []
    for atom_idx in atom_idx_group:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in atom_idx_group:
                result.append((atom_idx, neighbor.GetIdx()))
    return result
    

alt_atom = {bond_type: Chem.MolFromSmiles(atom_label).GetAtomWithIdx(0) for bond_type, atom_label in Fragment.bond_annotations.items()}

if __name__ == '__main__':
    pass