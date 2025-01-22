from rdkit import Chem
from .fragment_group import FragmentGroup, Fragment, FragmentBond, FragBondList
from .utils import *

def split_fragment(ori_fragment: Fragment, cut_bond_indices):
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



def merge_fragments(fragments: list[Fragment], merge_bond_poses, atom_id_list = None):
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

    return new_fragment, new_atom_maps