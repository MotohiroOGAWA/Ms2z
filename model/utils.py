from rdkit import Chem
import dill
from bidict import bidict


bond_priority = bidict({'-': 0, '=': 1, '#': 2, ':': 3})

def read_smiles(file_path, binary = None):
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

# #Mol->Mol (Error->None)
# def sanitize(mol, kekulize = True):
#     try:
#         smiles = get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
#         mol = get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
#     except:
#         mol = None
#     return mol



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
    smiles = Chem.MolToSmiles(mol, canonical=True)
    atom_order = list(map(int, mol.GetProp("_smilesAtomOutputOrder")[1:-2].split(",")))
    mol = Chem.MolFromSmiles(smiles)
    return mol, smiles, atom_order