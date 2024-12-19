from rdkit import Chem
import dill
from bidict import bidict
import copy


bond_priority = bidict({'-': 0, '=': 1, '#': 2, ':': 3})

def read_smiles(file_path, binary = None, split_smi = True):
    if binary is None:
        if file_path.endswith('.pkl'):
            binary = True
        else:
            binary = False

    if binary:
        with open(file_path, 'rb') as f:
            smiles = dill.load(f)
    else:
        if file_path.endswith('.sdf'):
            suppl = Chem.SDMolSupplier(file_path)
            smiles = [Chem.MolToSmiles(mol) for mol in suppl]

        with open(file_path, 'r') as f:
            if split_smi:
                smiles = [s.strip() for line in f for s in line.strip("\r\n ").split('.')]
            else:
                smiles = [line.strip("\r\n ").split()[0] for line in f]
            
    return smiles

def save_smiles(smiles, file_path, binary = None):
    if binary is None:
        if file_path.endswith('.pkl'):
            binary = True
        else:
            binary = False

    if binary:
        with open(file_path, 'wb') as f:
            dill.dump(smiles, f)
    else:
        with open(file_path, 'w') as f:
            for smi in smiles:
                f.write(smi + '\n')

def save_mol(mols, file_path):
    writer = Chem.SDWriter(file_path)
    for mol in mols:
        writer.write(mol)
    writer.close()

def load_mol(file_path):
    suppl = Chem.SDMolSupplier(file_path)
    mols = [mol for mol in suppl]
    return mols

#smiles->Mol
def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None: Chem.Kekulize(mol)
    return mol

#Mol->smiles
def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles = True)

#Mol->Mol (Error->None)
def sanitize(mol, kekulize = True):
    try:
        smiles = get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
        mol = get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
    except:
        mol = None
    return mol

def chem_bond_to_token(bond_type):
    if bond_type == Chem.BondType.SINGLE:
        bonding_type = "-"
    elif bond_type == Chem.BondType.DOUBLE:
        bonding_type = "="
    elif bond_type == Chem.BondType.TRIPLE:
        bonding_type = "#"
    elif bond_type == Chem.BondType.AROMATIC:
        bonding_type = ':'
    else:
        raise ValueError("Invalid bond type.")
    return bonding_type

def token_to_chem_bond(token):
    if token == "-":
        bond_type = Chem.BondType.SINGLE
    elif token == "=":
        bond_type = Chem.BondType.DOUBLE
    elif token == "#":
        bond_type = Chem.BondType.TRIPLE
    elif token == ":":
        bond_type = Chem.BondType.AROMATIC
    else:
        raise ValueError("Invalid bond token.")
    return bond_type

def token_to_num_bond(token, aromatic_as_half=False):
    """
    Converts a bond token to a numerical representation.

    Parameters:
        token (str): Bond token ('-', '=', '#', ':').
        aromatic_as_half (bool): If True, represents aromatic bonds (:) as 2.5 (float),
                                 and ensures all other bonds return as floats.

    Returns:
        int or float: Numerical representation of the bond.
    
    Raises:
        ValueError: If the provided token is invalid.
    """
    if token == "-":
        bond_type = 1.0 if aromatic_as_half else 1  # Single bond
    elif token == "=":
        bond_type = 2.0 if aromatic_as_half else 2  # Double bond
    elif token == "#":
        bond_type = 3.0 if aromatic_as_half else 3  # Triple bond
    elif token == ":":
        bond_type = 2.5 if aromatic_as_half else 4  # Aromatic bond
    else:
        raise ValueError("Invalid bond token.")  # Raise error for invalid tokens
    return bond_type

def get_atom_symbol(atom):
    """
    Returns a custom symbol for an atom based on its explicit hydrogens and radical state.
    
    - Normal atoms (no explicit hydrogens or radicals) are represented simply by their symbol (e.g., 'C').
    - Atoms with explicit hydrogens are represented in square brackets with 'H' and the number of hydrogens (e.g., '[CH2]').
    - Radical atoms are represented in square brackets without additional markers (e.g., '[C]').

    Args:
        atom (rdkit.Chem.Atom): An RDKit atom object.

    Returns:
        str: A string representing the atom symbol with explicit hydrogens and radicals, if any.
    """
    # Get the basic chemical symbol of the atom (e.g., 'C', 'O', 'N').
    symbol = atom.GetSymbol()
    
    # Get the number of unpaired electrons (radicals) on the atom.
    num_radicals = atom.GetNumRadicalElectrons()
    
    # Get the number of explicitly bonded hydrogen atoms.
    explicit_hs = atom.GetNumExplicitHs()
    
    # If the atom has no radicals or explicit hydrogens, return the basic symbol.
    if num_radicals == 0 and explicit_hs == 0:
        return symbol
    
    # Start creating a detailed symbol with the base chemical symbol.
    detailed_symbol = f"[{symbol}"
    
    # Append the number of explicit hydrogens if they exist.
    if explicit_hs > 0:
        detailed_symbol += f"H{explicit_hs}"
    
    # Radicals are acknowledged but do not add any extra markers.
    # (Radicals are implicitly represented by the presence of square brackets.)
    if num_radicals > 0:
        pass  # No additional markers for radicals.
    
    # Close the square brackets and return the detailed symbol.
    detailed_symbol += "]"
    return detailed_symbol

def get_atom_order(mol):
    """
    Extracts the canonical SMILES string and atom order from a molecule.

    Args:
        mol (Chem.Mol): An RDKit molecule object.

    Returns:
        tuple: A tuple containing:
            - str: The canonical SMILES string of the molecule.
            - list: A list of integers representing the atom order in the canonical SMILES.

    Notes:
        - The canonical SMILES is a unique representation of the molecule, ensuring consistent atom ordering.
        - The atom order is derived from the "_smilesAtomOutputOrder" property of the molecule,
          which is assumed to be precomputed and formatted as a comma-separated string of indices
          enclosed in brackets, e.g., "[0,1,2,...]".

    Raises:
        KeyError: If the molecule does not have the "_smilesAtomOutputOrder" property.
    """
    # smiles = Chem.MolToSmiles(mol, canonical=True)
    smiles = Chem.MolToSmiles(mol, canonical=True)
    atom_order = list(map(int, mol.GetProp("_smilesAtomOutputOrder")[1:-2].split(",")))
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    return mol, smiles, atom_order


def add_atom(rwmol, atom, new_atom, bond_type):
    if str(bond_type) == 'SINGLE':
        num = 1
    elif str(bond_type) == 'DOUBLE':
        num = 2
    elif str(bond_type) == 'TRIPLE':
        num = 3
    elif str(bond_type) == 'AROMATIC':
        # print("error in add_Hs 1")
        return rwmol
    else:
        print("error in add_Hs 2")
    
    new_atom_idx = rwmol.AddAtom(new_atom)
    rwmol.AddBond(atom.GetIdx(), new_atom_idx, bond_type)
    explicit_hs = atom.GetNumExplicitHs()
    rwmol = Chem.RWMol(Chem.AddHs(rwmol))
    atom_idx2 = [atom2.GetIdx() for atom2 in rwmol.GetAtoms() if atom.GetAtomMapNum() == atom2.GetAtomMapNum()][0]

    h_atoms = [neighbor for neighbor in rwmol.GetAtomWithIdx(atom_idx2).GetNeighbors() if neighbor.GetSymbol() == 'H']
    # for i in range(min(num, explicit_hs)):
    for i in range(min(num, len(h_atoms))):
        rwmol.RemoveAtom(h_atoms[i].GetIdx())
    
    aromatic_flags = {
        "atoms": {
            atom.GetAtomMapNum(): atom.GetIsAromatic() for atom in rwmol.GetAtoms()
        },
        "bonds": {
            (bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()): {
                "is_aromatic": bond.GetIsAromatic(),
                "bond_type": bond.GetBondType(),
            }
            for bond in rwmol.GetBonds()
        },
    }

    # Convert RWMol to a molecule without explicit hydrogens
    rwmol = Chem.RWMol(Chem.RemoveHs(rwmol))
    
    # Restore aromaticity and bond types
    for atom in rwmol.GetAtoms():
        # Use AtomMapNum if available, otherwise fall back to index
        key = atom.GetAtomMapNum() if atom.GetAtomMapNum() > 0 else atom.GetIdx()
        if key in aromatic_flags["atoms"]:
            atom.SetIsAromatic(aromatic_flags["atoms"][key])

    for bond in rwmol.GetBonds():
        # Use indices as bond keys
        begin_idx = bond.GetBeginAtom().GetAtomMapNum()
        end_idx = bond.GetEndAtom().GetAtomMapNum()
        bond_key = (begin_idx, end_idx)
        reverse_bond_key = (end_idx, begin_idx)  # Handle reversed bond keys
        
        # Restore bond properties
        if bond_key in aromatic_flags["bonds"]:
            bond_data = aromatic_flags["bonds"][bond_key]
        elif reverse_bond_key in aromatic_flags["bonds"]:
            bond_data = aromatic_flags["bonds"][reverse_bond_key]
        else:
            continue
        bond.SetBondType(bond_data["bond_type"])
        bond.SetIsAromatic(bond_data["is_aromatic"])
    
    return rwmol

def remove_atom(rwmol, removed_atom, neighbor_atom, bond_type):
    if str(bond_type) == 'SINGLE':
        num = 1
    elif str(bond_type) == 'DOUBLE':
        num = 2
    elif str(bond_type) == 'TRIPLE':
        num = 3
    elif str(bond_type) == 'AROMATIC':
        # print("error in add_Hs 1")
        return rwmol
    else:
        print("error in add_Hs 2")

    aromatic_flags = {
        "atoms": {
            atom.GetAtomMapNum(): atom.GetIsAromatic() for atom in rwmol.GetAtoms()
        },
        "bonds": {
            (bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()): {
                "is_aromatic": bond.GetIsAromatic(),
                "bond_type": bond.GetBondType(),
            }
            for bond in rwmol.GetBonds()
        },
    }

    rwmol.RemoveBond(removed_atom.GetIdx(), neighbor_atom.GetIdx())
    rwmol.RemoveAtom(removed_atom.GetIdx())

    for i in range(num):
        new_idx = rwmol.AddAtom(Chem.Atom(1))
        rwmol.GetAtomWithIdx(new_idx).SetAtomMapNum(0)
        rwmol.AddBond(new_idx, neighbor_atom.GetIdx(), Chem.BondType.SINGLE)

    # Restore aromaticity and bond types
    for atom in rwmol.GetAtoms():
        # Use AtomMapNum if available, otherwise fall back to index
        key = atom.GetAtomMapNum() if atom.GetAtomMapNum() > 0 else atom.GetAtomMapNum()
        if key in aromatic_flags["atoms"]:
            atom.SetIsAromatic(aromatic_flags["atoms"][key])

    for bond in rwmol.GetBonds():
        # Use indices as bond keys
        begin_idx = bond.GetBeginAtom().GetAtomMapNum()
        end_idx = bond.GetEndAtom().GetAtomMapNum()
        bond_key = (begin_idx, end_idx)
        reverse_bond_key = (end_idx, begin_idx)  # Handle reversed bond keys
        
        # Restore bond properties
        if bond_key in aromatic_flags["bonds"]:
            bond_data = aromatic_flags["bonds"][bond_key]
        elif reverse_bond_key in aromatic_flags["bonds"]:
            bond_data = aromatic_flags["bonds"][reverse_bond_key]
        else:
            continue
        bond.SetBondType(bond_data["bond_type"])
        bond.SetIsAromatic(bond_data["is_aromatic"])

    rwmol = Chem.RWMol(Chem.RemoveHs(rwmol))

    # Restore aromaticity and bond types
    for atom in rwmol.GetAtoms():
        # Use AtomMapNum if available, otherwise fall back to index
        key = atom.GetAtomMapNum() if atom.GetAtomMapNum() > 0 else atom.GetAtomMapNum()
        if key in aromatic_flags["atoms"]:
            atom.SetIsAromatic(aromatic_flags["atoms"][key])

    for bond in rwmol.GetBonds():
        # Use indices as bond keys
        begin_idx = bond.GetBeginAtom().GetAtomMapNum()
        end_idx = bond.GetEndAtom().GetAtomMapNum()
        bond_key = (begin_idx, end_idx)
        reverse_bond_key = (end_idx, begin_idx)  # Handle reversed bond keys
        
        # Restore bond properties
        if bond_key in aromatic_flags["bonds"]:
            bond_data = aromatic_flags["bonds"][bond_key]
        elif reverse_bond_key in aromatic_flags["bonds"]:
            bond_data = aromatic_flags["bonds"][reverse_bond_key]
        else:
            continue
        bond.SetBondType(bond_data["bond_type"])
        bond.SetIsAromatic(bond_data["is_aromatic"])

    return rwmol


#Valence adjustment by hydrogen addition after decomposition
def add_Hs(rwmol, a1, a2, bond):
    if str(bond.GetBondType()) == 'SINGLE':
        num = 1
    elif str(bond.GetBondType()) == 'DOUBLE':
        num = 2
    elif str(bond.GetBondType()) == 'TRIPLE':
        num = 3
    elif str(bond.GetBondType()) == 'AROMATIC':
        # print("error in add_Hs 1")
        return rwmol
    else:
        print("error in add_Hs 2")
        
    for i in range(num):
        new_idx = rwmol.AddAtom(Chem.Atom(1))
        rwmol.GetAtomWithIdx(new_idx).SetAtomMapNum(0)
        rwmol.AddBond(new_idx, a1.GetIdx(), Chem.BondType.SINGLE)
        if a2 != None:
            new_idx = rwmol.AddAtom(Chem.Atom(1))
            rwmol.GetAtomWithIdx(new_idx).SetAtomMapNum(0)
            rwmol.AddBond(new_idx, a2.GetIdx(), Chem.BondType.SINGLE)
    return rwmol

def remove_Hs(rwmol, a1, a2, bond_type):
    try:
        if str(bond_type) == 'SINGLE':
            num = 1
        elif str(bond_type) == 'DOUBLE':
            num = 2
        elif str(bond_type) == 'TRIPLE':
            num = 3
        elif str(bond_type) == 'AROMATIC':
            print("error in remove_Hs 1")
        else:
            print("error in remove_Hs 2")
    except:
        if bond_type == 0:
            num = 1
        elif bond_type == 1:
            num = 2
        elif bond_type == 2:
            num = 3
        else:
            raise
    rwmol = Chem.AddHs(rwmol)
    rwmol = Chem.RWMol(rwmol)
    #Set hydrogen maps for connected atoms
    h_map1 = 2000000
    h_map2 = 3000000
    f_h_map1 = copy.copy(h_map1)
    f_h_map2 = copy.copy(h_map2)
    for b in rwmol.GetBonds():
        s_atom = b.GetBeginAtom()
        e_atom = b.GetEndAtom()
        if (e_atom.GetIdx() == a1.GetIdx()) and (s_atom.GetSymbol() == 'H'):
            s_atom.SetAtomMapNum(h_map1)
            h_map1 += 1
        elif (s_atom.GetIdx() == a1.GetIdx()) and (e_atom.GetSymbol() == 'H'):
            e_atom.SetAtomMapNum(h_map1)
            h_map1 += 1
        elif (e_atom.GetIdx() == a2.GetIdx()) and (s_atom.GetSymbol() == 'H'):
            s_atom.SetAtomMapNum(h_map2)
            h_map2 += 1
        elif (s_atom.GetIdx() == a2.GetIdx()) and (e_atom.GetSymbol() == 'H'):
            e_atom.SetAtomMapNum(h_map2)
            h_map2 += 1
    for i in range(num):
        try:
            for atom in rwmol.GetAtoms():
                if atom.GetAtomMapNum() == f_h_map1 + i:
                    rwmol.RemoveAtom(atom.GetIdx())
                    break
            for atom in rwmol.GetAtoms():
                if atom.GetAtomMapNum() == f_h_map2 + i:
                    rwmol.RemoveAtom(atom.GetIdx())
                    break
        except:
            print("Remove Hs times Error!!")
            raise
    rwmol = rwmol.GetMol()
    rwmol = sanitize(rwmol, kekulize = False)
    rwmol = Chem.RemoveHs(rwmol)
    rwmol = Chem.RWMol(rwmol)
    return rwmol
