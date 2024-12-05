from rdkit import Chem
import copy
from collections import defaultdict

try:
    from utils import *
except:
    from .utils import *



class Fragment:
    # Map bond types to their respective annotations
    bond_annotations = {'-': '[Nh]', '=': '[Og]', '#': '[Ts]'}

    def __init__(self, smiles, bond_infoes, atom_map=None):
        mol, smiles, bond_infoes, atom_map = Fragment.normalize_frag_info(smiles, bond_infoes, atom_map)

        self.mol = mol
        self.smiles = smiles
        self.bond_infoes = bond_infoes
        self.atom_map = atom_map
        

    def __str__(self):
        output = [self.smiles] + [item for atom_idx_bond_type in self.bond_infoes for item in atom_idx_bond_type]
        return str(tuple(output))

    def __repr__(self):
        return str(self.__str__())
    
    @staticmethod
    def normalize_frag_info(smiles, bond_infoes, atom_map = None):
        mol = Chem.MolFromSmiles(smiles)
        mol_with_alt, smiles_with_alt, atom_map_with_alt = Fragment._add_bond_annotations(mol, bond_infoes, atom_map)
        normalized_mol, normalized_smiles, normalized_atom_map, normalized_bond_infoes = Fragment._remove_bond_annotations(mol_with_alt, atom_map_with_alt)
        return normalized_mol, normalized_smiles, normalized_bond_infoes, normalized_atom_map

    @staticmethod
    def _add_bond_annotations(mol, bond_infoes, atom_indices = None):
        """
        Adds bond annotations to a molecule based on the bond information provided.

        Args:
            mol (Chem.Mol): RDKit molecule object.
            bond_infoes (list): List of tuples containing atom indices and bond types.
            atom_indices (list): List of atom indices corresponding to the bond information.

        Returns:
            Chem.Mol: RDKit molecule object with bond annotations.
        """
        if atom_indices is None:
            atom_indices = list(range(mol.GetNumAtoms()))
        
        map_num_to_ori_atom_idx = {}
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(i)
            map_num_to_ori_atom_idx[i] = atom_indices[i]

        rwmol = Chem.RWMol(mol)
        atom_idx_offfset = mol.GetNumAtoms()
        for i, bond_info in enumerate(bond_infoes):
            bond_pos, bond_type = bond_info

            new_atom = Chem.Atom(alt_atom[bond_type])
            new_atom.SetAtomMapNum(atom_idx_offfset + i)
            new_atom_idx = rwmol.AddAtom(new_atom)
            rwmol.AddBond(bond_pos, new_atom_idx,
            Fragment.token_to_chem_bond(bond_type))
            map_num_to_ori_atom_idx[atom_idx_offfset + i] = - (i + 1)
        
        new_mol = rwmol.GetMol()
        atom_map = {}
        for i, atom in enumerate(new_mol.GetAtoms()):
            atom_map[i] = atom.GetAtomMapNum()
        for atom in new_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        
        new_mol, new_smiles, atom_order = get_atom_order(new_mol)
        new_atom_map = {map_num_to_ori_atom_idx[atom_map[idx]]: i for i, idx in enumerate(atom_order)}
        new_atom_map = [k for k, v in sorted(new_atom_map.items(), key=lambda item: item[1])]

        return new_mol, new_smiles, new_atom_map
    
    @staticmethod
    def _remove_bond_annotations(mol, atom_indices=None):
        """
        Removes bond annotations from a molecule.

        Args:
            mol (Chem.Mol): RDKit molecule object with bond annotations.

        Returns:
            Chem.Mol: RDKit molecule object without bond annotations.
        """
        if atom_indices is None:
            atom_indices = list(range(mol.GetNumAtoms()))

        atom_map_to_idx = {}
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(atom.GetIdx())
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
            alt_atom_idx = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == alt_atom_map_num][0]
            atom_idx = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == atom_map_num][0]
        # for (atom_idx, alt_atom_idx) in bond_atom_idx_pairs:
            bond = rwmol.GetBondBetweenAtoms(alt_atom_idx, atom_idx)
            if bond:
                rwmol.RemoveBond(alt_atom_idx, atom_idx)
                rwmol.RemoveAtom(alt_atom_idx)
            
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

        bond_infoes = []
        for atom_map_num, bond_token_list in bond_atom_indices.items():
            for bond_token in bond_token_list:
                bond_infoes.append((atom_order.index(map_num_to_pre_order[atom_map_num]), bond_token))
        bond_infoes = sorted(bond_infoes, key=lambda x: (x[0], bond_priority[x[1]]))

        return new_mol, smiles, new_atom_map, bond_infoes


    
    @staticmethod
    def split_molecule(mol, min_non_ring_neighbors=0):
        """
        Splits a molecule into fragments based on functional groups and connectivity,
        while tracking broken bond information and ensuring proper handling of ring and non-ring regions.

        Args:
            mol (Chem.Mol): Input RDKit molecule object.
            min_non_ring_neighbors (int, optional): Minimum number of non-ring neighbors
                an atom must have for the bond to be split. Defaults to 0.

        Returns:
            tuple:
                - count_labels (list): List of tuples containing fragment SMILES,
                bond type, and positional information.
                - fragments (list): List of RDKit molecule objects for each fragment.
                - atom_tokens (list): List of atom tokens from the original molecule.
        """
        mol = Chem.rdmolops.RemoveHs(mol)  # Remove explicit hydrogens
        atom_tokens = Fragment.get_monoatomic_tokens(mol)  # Atom tokens for later reference

        # Create a new editable molecule
        new_mol = Chem.RWMol(mol)

        # Assign AtomMapNum to track original atom indices
        for atom in new_mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())

        sep_sets = []  # List to store bonds selected for splitting
        sep_sets = self.separate_functional(mol)

        # Identify bonds to split based on ring membership and connectivity
        for bond in mol.GetBonds():
            if bond.IsInRing():  # Skip bonds inside rings
                continue
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()

            # If both atoms are inside a ring, split the bond
            if a1.IsInRing() and a2.IsInRing():
                if not (((a1.GetIdx(), a2.GetIdx()) in sep_sets) or
                    ((a2.GetIdx(), a1.GetIdx()) in sep_sets)):
                    sep_sets.append((a1.GetIdx(), a2.GetIdx()))
            # Split if one atom is in a ring and the other is highly connected
            elif ((a1.IsInRing() and a2.GetDegree() > min_non_ring_neighbors) 
                  or (a2.IsInRing() and a1.GetDegree() > min_non_ring_neighbors)):
                if not (((a1.GetIdx(), a2.GetIdx()) in sep_sets) or
                    ((a2.GetIdx(), a1.GetIdx()) in sep_sets)):
                    sep_sets.append((a1.GetIdx(), a2.GetIdx()))

        # Dictionary to map original atoms to split indices
        atommap_dict = defaultdict(list)
        sep_idx = 1
        atommap_dict = defaultdict(list) #key->AtomIdx, value->sep_idx (In the whole compound before decomposition)
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if ((a1.GetIdx(),a2.GetIdx()) in sep_sets) or ((a2.GetIdx(),a1.GetIdx()) in sep_sets):
                a1map = new_mol.GetAtomWithIdx(a1.GetIdx()).GetAtomMapNum()
                a2map = new_mol.GetAtomWithIdx(a2.GetIdx()).GetAtomMapNum()
                atommap_dict[a1map].append(sep_idx)
                atommap_dict[a2map].append(sep_idx)
                new_mol = add_Hs(new_mol, a1, a2, bond)
                new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())
                sep_idx += 1
        for i in list(atommap_dict.keys()):
            atommap_dict[i] = sorted(atommap_dict[i])  
        for i in list(atommap_dict.keys()):
            if atommap_dict[i] == []:
                atommap_dict.pop(i)
        new_mol = new_mol.GetMol()
        new_mol = sanitize(new_mol, kekulize = False)
        new_smiles = Chem.MolToSmiles(new_mol)
        fragment_mols = [Chem.MolFromSmiles(fragment) for fragment in new_smiles.split('.')]
        fragment_mols = [sanitize(fragment, kekulize = False) for fragment in fragment_mols]
        
        fragment_labels = []
        fragment_atom_mapping = []  # To track atom indices corresponding to each fragment

        for i, fragment_mol in enumerate(fragment_mols):
            order_list = [] #Stores join orders in the substructures
            fragment_label = []
            frag_atom_indices = []  # To store original atom indices for this fragment
            frag_mol = copy.deepcopy(fragment_mol)
            for atom in frag_mol.GetAtoms():
                frag_mol.GetAtomWithIdx(atom.GetIdx()).SetAtomMapNum(0)
            frag_smi = Chem.MolToSmiles(sanitize(frag_mol, kekulize = False))
            #Fix AtomIdx as order changes when AtomMap is deleted.
            atom_order = list(map(int, frag_mol.GetProp("_smilesAtomOutputOrder")[1:-2].split(",")))
            a_dict = {}
            for i, atom in enumerate(fragment_mol.GetAtoms()):
                amap = atom.GetAtomMapNum()
                a_dict[i] = amap
                if amap in list(atommap_dict.keys()):
                    order_list.append(atommap_dict[amap])

            for order in atom_order:
                frag_atom_indices.append(a_dict[order])

            order_list = sorted(order_list)
            fragment_label.append(frag_smi)
            for atom in fragment_mol.GetAtoms():
                amap = atom.GetAtomMapNum()
                if amap in list(atommap_dict.keys()):
                    for seq_idx in atommap_dict[amap]:
                        for amap2 in list(atommap_dict.keys()):
                            if (seq_idx in atommap_dict[amap2]) and (amap != amap2):
                                bond = mol.GetBondBetweenAtoms(amap, amap2)
                                bond_type = bond.GetBondType()
                                bond_type_str = ""
                                if bond_type == Chem.BondType.SINGLE:
                                    bond_type_str = "-"
                                elif bond_type == Chem.BondType.DOUBLE:
                                    bond_type_str = "="
                                elif bond_type == Chem.BondType.TRIPLE:
                                    bond_type_str = "#"
                                elif bond_type == Chem.BondType.AROMATIC:
                                    bond_type_str = ":"

                                fragment_label.append(atom_order.index(atom.GetIdx()))
                                fragment_label.append(bond_type_str)
                                # count_label.append(order_list.index(atommap_dict[amap]) + 1)

            fragment_label, frag_atom_indices = normalize_bond_info(fragment_label, frag_atom_indices)
            fragment_atom_mapping.append(frag_atom_indices)
            fragment_labels.append(tuple(fragment_label))
        return fragment_labels, fragment_mols, atom_tokens, fragment_atom_mapping
    
    @staticmethod
    def split_to_monoatomic_fragment(mol):
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
                    bond_token = Fragment.chem_bond_to_token(bond_type)
                    bond_tokens.append(bond_token)

                bond_tokens = sorted(bond_tokens, key=lambda x: bond_priority[x])
                
            bond_infoes = [(0, bond_token) for bond_token in bond_tokens]

            fragments.append(Fragment(Chem.MolToSmiles(mol), bond_infoes))

        return fragments
    
    @staticmethod
    def get_monoatomic_tokens(mol):
        tokens = set()
        for atom in mol.GetAtoms():
            atom_symbol = get_atom_symbol(atom)
    
    @staticmethod
    def to_tuple(fragment):
        pass

    @staticmethod
    def chem_bond_to_token(bond_type):
        if bond_type == Chem.BondType.SINGLE:
            bond_token = "-"
        elif bond_type == Chem.BondType.DOUBLE:
            bond_token = "="
        elif bond_type == Chem.BondType.TRIPLE:
            bond_token = "#"
        elif bond_type == Chem.BondType.AROMATIC:
            bond_token = ':'
        else:
            raise ValueError("Invalid bond type.")
        return bond_token

    @staticmethod
    def token_to_chem_bond(bond_token):
        if bond_token == "-":
            bond_type = Chem.BondType.SINGLE
        elif bond_token == "=":
            bond_type = Chem.BondType.DOUBLE
        elif bond_token == "#":
            bond_type = Chem.BondType.TRIPLE
        elif bond_token == ":":
            bond_type = Chem.BondType.AROMATIC
        else:
            raise ValueError("Invalid bond token.")
        return bond_type


alt_atom = {bond_type: Chem.MolFromSmiles(atom_label).GetAtomWithIdx(0) for bond_type, atom_label in Fragment.bond_annotations.items()}

if __name__ == '__main__':


    smiles = 'C([CH2])OC=C(C#N)C(C)C'
    fragment = Fragment(smiles, bond_infoes=[(1, '-'), (0, '-'), (0, '-'), (8, '#'), (9, '=')])
    print(fragment)