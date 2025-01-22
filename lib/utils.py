from rdkit import Chem
from rdkit.Chem import rdFMCS
import numpy as np
import dill
from bidict import bidict
import copy
from collections import Counter, defaultdict


bond_priority = bidict({'-': 0, '=': 1, '#': 2, ':': 3})

def read_smiles(file_path, binary = None, split_smi = True, duplicate = False):
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
    
    if not duplicate:
        smiles = list(set(smiles))
            
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

def num_bond_to_token(num, aromatic_as_half=False):
    """
    Converts a numerical representation of a bond to a token.

    Parameters:
        num (int or float): Numerical representation of a bond.
        aromatic_as_half (bool): If True, represents aromatic bonds (2.5) as ':',
                                 and ensures all other bonds return as strings.

    Returns:
        str: Bond token ('-', '=', '#', ':').

    Raises:
        ValueError: If the provided numerical bond is invalid.
    """
    if aromatic_as_half:
        if num == 1.0:
            bond_token = "-"
        elif num == 2.0:
            bond_token = "="
        elif num == 3.0:
            bond_token = "#"
        elif num == 2.5:
            bond_token = ":"
        else:
            raise ValueError("Invalid numerical bond.")
    else:
        if num == 1:
            bond_token = "-"
        elif num == 2:
            bond_token = "="
        elif num == 3:
            bond_token = "#"
        elif num == 4:
            bond_token = ":"
        else:
            raise ValueError("Invalid numerical bond.")
    return bond_token

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


def frag_to_joint_list(frag_tuple):
    atom_idx_to_bond_counter = defaultdict(lambda: [0, 0, 0])
    for atom_idx, bond_token in zip(frag_tuple[1::2], frag_tuple[2::2]):
        atom_idx_to_bond_counter[atom_idx][token_to_num_bond(bond_token)-1] += 1
    joint_list = [(atom_idx, bond_counter[0], bond_counter[1], bond_counter[2]) for atom_idx, bond_counter in atom_idx_to_bond_counter.items()]
    return joint_list


def get_ring_groups(mol):
    """
    Group the ring structures in a molecule and return them.
    Rings that share common atoms are merged into the same group.

    Parameters:
        mol (Chem.Mol): RDKit molecule object

    Returns:
        list[set]: A list of sets, where each set contains the atom indices of a ring group.
    """
    # Retrieve ring information from the molecule
    ring_info = mol.GetRingInfo()
    all_rings = [set(ring) for ring in ring_info.AtomRings()]  # Convert each ring to a set

    # Merge connected ring information
    ring_groups = []  # List to store ring groups
    visited_rings = set()  # Set to track visited ring indices

    for i, ring in enumerate(all_rings):
        if i in visited_rings:
            continue

        # Create a new group and start exploration
        current_group = ring.copy()
        visited_rings.add(i)
        queue = [i]

        while queue:
            current_ring_idx = queue.pop(0)
            current_ring = all_rings[current_ring_idx]

            # Check for connections with other rings
            for j, next_ring in enumerate(all_rings):
                if j not in visited_rings and current_ring.intersection(next_ring):
                    visited_rings.add(j)
                    queue.append(j)
                    current_group.update(next_ring)

        # Add the completed group
        ring_groups.append(current_group)

    return ring_groups

        

def count_atoms_in_molecule(mol, ignore_hydrogen=True):
    """
    Count the number of each type of atom in a molecule, optionally ignoring hydrogen atoms.

    Args:
        mol (rdkit.Chem.Mol): Input molecule.
        ignore_hydrogen (bool, optional): If True, hydrogen atoms (H) will be ignored. Defaults to True.

    Returns:
        Counter: A Counter object with atom symbols as keys and their counts as values.
    """
    # Get a list of atom symbols
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    
    # Optionally filter out hydrogen atoms
    if ignore_hydrogen:
        atom_symbols = [symbol for symbol in atom_symbols if symbol != "H"]
    
    # Return a Counter object of atom counts
    return Counter(atom_symbols)

def count_bonds_in_molecule(mol):
    """
    Count the number of each type of bond in a molecule.

    Args:
        mol (rdkit.Chem.Mol): Input molecule.

    Returns:
        Counter: A Counter object with bond types as keys and their counts as values.
    """
    # Get a list of bond types
    bond_types = [chem_bond_to_token(bond.GetBondType()) for bond in mol.GetBonds()]
    
    # Return a Counter object of bond counts
    return Counter(bond_types)

def softmax(x):
    """
    Compute softmax values for a list or array.
    """
    exp_x = np.exp(np.max(x) - x)  # Numerical stability adjustment
    return exp_x / exp_x.sum()

def compute_weights_with_softmax(raw_weights):
    """
    Adjust raw weights using softmax to emphasize rare atoms.

    Args:
        raw_weights (dict): Raw weights for atoms.

    Returns:
        dict: Adjusted weights after applying softmax.
    """
    symbols = list(raw_weights.keys())
    raw_values = np.array(list(raw_weights.values()))
    adjusted_values = softmax(-raw_values)  # Invert values to prioritize low-frequency atoms
    return dict(zip(symbols, adjusted_values))

def mask_atoms_with_custom_symbol(mol, match_indices, symbol):
    """
    Masks specified atoms in a molecule by replacing them with a custom atomic symbol.
    """
    editable_mol = Chem.EditableMol(mol)
    for idx in sorted(match_indices, reverse=True):  # Process indices in reverse order
        new_atom = Chem.Atom(symbol)
        editable_mol.ReplaceAtom(idx, new_atom)
    return editable_mol.GetMol()

def find_mcs_and_mask_custom(mol1, mol2, weights=None, iterations=2):
    """
    Iteratively finds the Maximum Common Substructure (MCS) between two molecules
    and masks the matched atoms with custom symbols for further analysis.

    Args:
        mol1 (rdkit.Chem.Mol): First molecule.
        mol2 (rdkit.Chem.Mol): Second molecule.
        weights (dict or None): Dictionary of atom weights. If None, all weights are treated as 1.0.
        iterations (int): Maximum number of iterations for finding MCS.

    Returns:
        float: Weighted total number of atoms in the common substructures across all iterations.
    """
    total_common_weighted_atoms = 0.0
    masked_mol1 = mol1
    masked_mol2 = mol2

    for _ in range(iterations):
        # Find the Maximum Common Substructure (MCS)
        mcs = rdFMCS.FindMCS([masked_mol1, masked_mol2])
        if mcs.numAtoms == 0:
            break  # Stop if no common structure is found

        # Get the MCS molecule and atom matches
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        match1 = masked_mol1.GetSubstructMatch(mcs_mol)
        match2 = masked_mol2.GetSubstructMatch(mcs_mol)

        if not match1 or not match2:
            break

        # Calculate weighted common atoms for the current MCS
        for atom_idx in match1:  # Iterate over matched atoms in mol1
            atom_symbol = masked_mol1.GetAtomWithIdx(atom_idx).GetSymbol()
            weight = weights.get(atom_symbol, 1.0) if weights else 1.0
            total_common_weighted_atoms += 1.0 / weight

        # Mask matched atoms in each molecule with custom symbols
        masked_mol1 = mask_atoms_with_custom_symbol(masked_mol1, match1, "Nh")
        masked_mol2 = mask_atoms_with_custom_symbol(masked_mol2, match2, "Og")

    return total_common_weighted_atoms

def calculate_mcs_similarity(mol1, mol2, weights=None):
    """
    Calculate similarity between two molecules using MCS and weighted atom counts.

    Args:
        mol1 (rdkit.Chem.Mol): First molecule.
        mol2 (rdkit.Chem.Mol): Second molecule.
        weights (dict or None): Dictionary of atom weights. If None, all weights are treated as 1.0.
                                Example:
                                    raw_weights = {"C": 0.9, "Br": 0.0003}
                                    weights = compute_weights_with_softmax(raw_weights)
                                    Resulting weights:
                                    {
                                        "C": 0.28911215136850005,  # Lower weight for common atoms
                                        "Br": 0.7108878486314999   # Higher weight for rare atoms
                                    }

    Returns:
        float: Adjusted similarity score.
    """
    # Get the total weighted number of atoms in each molecule
    weighted_total_atoms_mol1 = sum(
        1.0 / weights.get(atom.GetSymbol(), 1.0) if weights else 1.0
        for atom in mol1.GetAtoms()
    )
    weighted_total_atoms_mol2 = sum(
        1.0 / weights.get(atom.GetSymbol(), 1.0) if weights else 1.0
        for atom in mol2.GetAtoms()
    )

    # Find the total weighted number of common atoms using the iterative MCS function
    weighted_common_atoms = find_mcs_and_mask_custom(mol1, mol2, weights, iterations=10)

    # Calculate the total weighted number of atoms in the union of both molecules
    weighted_union_atoms = weighted_total_atoms_mol1 + weighted_total_atoms_mol2 - weighted_common_atoms

    # Calculate similarity as the ratio of weighted common atoms to weighted union atoms
    similarity = weighted_common_atoms / weighted_union_atoms
    return similarity
