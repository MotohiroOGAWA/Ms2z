from .utils import token_to_chem_bond, token_to_num_bond, bond_priority


class FragmentBond:
    def __init__(self, atom_idx, bond_token):
        self.id = -1
        self.atom_idx = atom_idx
        self.token = bond_token
        self.type = token_to_chem_bond(bond_token)
        self.num = token_to_num_bond(bond_token, aromatic_as_half=False)

    def __str__(self):
        return f"({self.atom_idx}, {self.token})"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return self.atom_idx == other.atom_idx and self.token == other.bond_token
    
    def __iter__(self):
        return iter((self.atom_idx, self.token))

class FragBondList:
    def __init__(self, bond_info=None):
        """
        bond_info: list of tuples [(atom_idx, bond_token), ...] or list of FragmentBond instances
        """
        if bond_info:
            self.bonds = [info if isinstance(info, FragmentBond) else FragmentBond(info[0], info[1]) for info in bond_info]
            self._sort_and_reassign_ids()
        else:
            self.bonds = []
    
    def __str__(self):
        return str([bond for bond in self.bonds])
    
    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.bonds)
    
    def __iter__(self):
        return iter(self.bonds)
    
    def __getitem__(self, key):
        return self.bonds[key]
    
    def _sort_and_reassign_ids(self):
        self.bonds = sorted(self.bonds, key=lambda x: (x.atom_idx, bond_priority[x.token]))
        for idx, bond in enumerate(self.bonds):
            bond.id = idx

    def add(self, fragment_bond):
        if isinstance(fragment_bond, FragmentBond):
            self.bonds.append(fragment_bond)
            self._sort_and_reassign_ids()
        else:
            raise ValueError("Only FragmentBond instances can be added.")
        
    def remove(self, fragment_bond):
        for tgt_frag_bond in self.bonds:
            if tgt_frag_bond == fragment_bond:
                self.bonds.remove(tgt_frag_bond)
                self._sort_and_reassign_ids()
                return True
        return False
    
    def get_bond_ids(self, atom_idx, bond_token):
        return [bond.id for bond in self.bonds if bond.atom_idx == atom_idx and bond.token == bond_token]
    
    def get_bond_id_and_token(self, atom_idx):
        bond_ids = [bond.id for bond in self.bonds if bond.atom_idx == atom_idx]
        bond_tokens = [bond.token for bond in self.bonds if bond.atom_idx == atom_idx]
        return bond_ids, bond_tokens
    
    def tolist(self):
        return [(bond.atom_idx, bond.token) for bond in self.bonds]
        
    
    @property
    def tokens(self):
        """Returns a list of bond tokens."""
        return [bond.token for bond in self.bonds]

    
    