from rdkit import Chem
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)  # Only show critical errors, suppress warnings and other messages



def read_smiles_file(file_path):
    with open(file_path, 'r') as f:
        smiles_list = f.read().splitlines()
    
    return smiles_list

def process_smiles_list_to_unique_smiles_chunk(smiles_chunk):
    """
    Process a chunk of SMILES strings to generate unique canonical SMILES.

    Parameters:
        smiles_chunk (list): A list of SMILES strings.

    Returns:
        list: Unique canonical SMILES strings.
    """
    unique_smiles = set()
    for smiles in smiles_chunk:
        try:
            mol = Chem.MolFromSmiles(smiles)
            smi = Chem.MolToSmiles(mol, canonical=True)
            unique_smiles.add(smi)
        except Exception:
            pass
    return list(unique_smiles)

def smiles_list_to_unique_smiles(smiles_list, chunk_size=1000, num_workers=None):
    """
    Process a list of SMILES strings in parallel to extract unique canonical SMILES.

    Parameters:
        smiles_list (list): List of SMILES strings.
        chunk_size (int): Size of each chunk for processing (default: 1000).
        num_workers (int): Number of parallel workers (default: None, which uses the number of CPUs).

    Returns:
        list: Unique canonical SMILES strings.
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    # Split SMILES list into chunks
    chunks = [smiles_list[i:i + chunk_size] for i in range(0, len(smiles_list), chunk_size)]
    
    unique_smiles = set()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_smiles_list_to_unique_smiles_chunk, chunk) for chunk in chunks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing SMILES in parallel", unit="chunk"):
            result = future.result()
            unique_smiles.update(result)

    return list(unique_smiles)


def process_smiles_list_to_canonical_mapping_chunk(smiles_chunk):
    """
    Process a chunk of SMILES strings to generate a mapping of original to canonical SMILES.

    Parameters:
        smiles_chunk (list): A list of SMILES strings.

    Returns:
        dict: A mapping of original SMILES to canonical SMILES.
    """
    canonical_mapping = {}
    for smiles in smiles_chunk:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                canonical_mapping[smiles] = canonical_smiles
        except Exception:
            canonical_mapping[smiles] = None  # Mark invalid SMILES as None
    return canonical_mapping

def smiles_list_to_canonical_mapping(smiles_list, chunk_size=1000, num_workers=None):
    """
    Process a list of SMILES strings in parallel to create a mapping of original to canonical SMILES.

    Parameters:
        smiles_list (list): List of SMILES strings.
        chunk_size (int): Size of each chunk for processing (default: 1000).
        num_workers (int): Number of parallel workers (default: None, which uses the number of CPUs).

    Returns:
        tuple: A tuple containing:
            - dict: A mapping of original SMILES to canonical SMILES.
            - list: The canonical SMILES strings in the same order as the input list.
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    # Split SMILES list into chunks
    chunks = [smiles_list[i:i + chunk_size] for i in range(0, len(smiles_list), chunk_size)]

    canonical_mapping = {}
    canonical_smiles_list = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_smiles_list_to_canonical_mapping_chunk, chunk) for chunk in chunks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing SMILES in parallel", unit="chunk"):
            result = future.result()
            canonical_mapping.update(result)

    # Generate the ordered list of canonical SMILES
    for smiles in tqdm(smiles_list, desc="Convert to canonical SMILES list", unit="SMILES"):
        if smiles in canonical_mapping:
            canonical_smiles_list.append(canonical_mapping[smiles])
        else:
            canonical_smiles_list.append(None)

    return canonical_smiles_list

