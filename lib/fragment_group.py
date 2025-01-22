from collections import defaultdict
from typing import Iterator
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
        strings = [f'{fragment.id}:'+ str(tuple(fragment)) + ' --> ' + str(dict(sorted(self.get_neighbors(fragment.id).items(), key=lambda x: x[0]))) for fragment in self.fragments]
        return '\n'.join(strings)
    
    def __repr__(self):
        return self.__str__()

        
    def __len__(self):
        return len(self.fragments)
    
    def __iter__(self) -> Iterator[Fragment]:
        return iter(self.fragments)
    
    def __getitem__(self, key) -> Fragment:
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
        """
        Retrieve all neighbors of a specified fragment.

        This function returns a dictionary of all neighboring fragments for the specified fragment index.
        The dictionary keys represent the bond positions in the current fragment, and the values are tuples 
        containing the neighboring fragment index and the bond position in the neighboring fragment.

        Args:
            frag_idx (int): The index of the fragment for which neighbors are retrieved.

        Returns:
            dict: A dictionary where the keys are bond positions in the current fragment (`s_bond_pos`),
                and the values are tuples (`e_frag_idx`, `e_bond_pos`), where:
                - `e_frag_idx` (int): The index of the neighboring fragment.
                - `e_bond_pos` (int): The bond position in the neighboring fragment.
        """
        return {s_bond_pos: (e_frag_idx, e_bond_pos) for s_bond_pos, (e_frag_idx, e_bond_pos) in self._neighbors[frag_idx].items()}

    
    def get_neighbor(self, frag_idx, bond_pos):
        """
        Retrieve the specific neighbor connected to a given bond position.

        This function returns the fragment index and bond position of the neighbor connected to a 
        specified bond position in the current fragment.

        Args:
            frag_idx (int): The index of the fragment for which the neighbor is retrieved.
            bond_pos (int): The bond position in the fragment from which the neighbor is connected.

        Returns:
            tuple: A tuple containing:
                - `neighbor_frag_idx` (int): The index of the neighboring fragment.
                - `neighbor_bond_pos` (int): The bond position in the neighboring fragment.
        """
        if bond_pos not in self._neighbors[frag_idx]:
            return None, None
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
