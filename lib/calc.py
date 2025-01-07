from rdkit import Chem
from rdkit.Chem import MACCSkeys
import torch

from rdkit import Chem
from rdkit.Chem import MACCSkeys
import torch

def smiles_to_fp_tensor(smiles_list):
    """
    Convert a list of SMILES strings to a fingerprint tensor and dynamically choose the device (CUDA or CPU).

    Parameters:
        smiles_list (list of str): List of SMILES strings.

    Returns:
        torch.Tensor: Tensor of shape (len(smiles_list), 166), containing the fingerprints.
    """
    # Automatically determine the device (CUDA or CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize an empty list to store fingerprints
    fp_list = []

    for smiles in smiles_list:
        try:
            # Convert SMILES to an RDKit Mol object
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")

            # Generate the MACCS fingerprint
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)

            # Convert the fingerprint to a list of integers
            fp_list.append(list(maccs_fp))
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            # Append a zero vector if SMILES conversion fails
            fp_list.append([0] * 167)

    # Convert the list of fingerprints to a PyTorch tensor and move to the determined device
    fp_tensor = torch.tensor(fp_list, dtype=torch.float32, device=device)

    return fp_tensor

def compute_tanimoto_similarity(fp_tensor, device=None):
    """
    Compute the Tanimoto similarity matrix for a set of fingerprints.

    Parameters:
        fp_tensor (torch.Tensor): A binary tensor of shape (datasize, num_bits),
                                  where each row represents a fingerprint.

    Returns:
        torch.Tensor: A tensor of shape (datasize, datasize) containing Tanimoto similarities.
    """
    if device is not None:
        fp_tensor = fp_tensor.to(device)

    # Calculate the AND (intersection) and OR (union) for all pairs
    intersection = torch.mm(fp_tensor, fp_tensor.T)
    fp_sums = fp_tensor.sum(dim=1).unsqueeze(1)  # Precompute sums
    union = fp_sums + fp_sums.T - intersection

    # Compute Tanimoto similarity
    tanimoto_matrix = intersection / union

    # Replace NaN values with 0
    tanimoto_matrix = torch.nan_to_num(tanimoto_matrix, nan=0.0)

    # Return the result to CPU for further processing if needed
    return tanimoto_matrix

def compute_cosine_similarity(fp_tensor, device=None):
    """
    Compute the Cosine similarity matrix for a set of fingerprints.

    Parameters:
        fp_tensor (torch.Tensor): A tensor of shape (datasize, num_bits),
                                  where each row represents a fingerprint.
        device (str or torch.device, optional): The device to use ('cuda' or 'cpu').

    Returns:
        torch.Tensor: A tensor of shape (datasize, datasize) containing Cosine similarities.
    """
    if device is not None:
        fp_tensor = fp_tensor.to(device)

    # Normalize each row to have unit length
    fp_norm = fp_tensor / fp_tensor.norm(dim=1, keepdim=True)

    # Compute the Cosine similarity matrix
    cosine_similarity_matrix = torch.mm(fp_norm, fp_norm.T)

    # Replace NaN values with 0
    cosine_similarity_matrix = torch.nan_to_num(cosine_similarity_matrix, nan=0.0)

    return cosine_similarity_matrix

if __name__ == "__main__":
    # Example usage
    smiles_list = ["CCO", "c1ccccc1O", "C1=CC=CC=C1", "INVALID_SMILES"]

    # Generate the fingerprint tensor
    fp_tensor = smiles_to_fp_tensor(smiles_list)
    tanimoto_matrix = compute_tanimoto_similarity(fp_tensor)
    cosine_matrix = compute_cosine_similarity(fp_tensor)

    # Print the device and the resulting tensor
    print(f"Tensor is on device: {fp_tensor.device}")
    print(fp_tensor)

