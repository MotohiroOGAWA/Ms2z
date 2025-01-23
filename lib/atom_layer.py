from rdkit import Chem
from rdkit.Chem import rdchem
import torch

PERIODIC_TABLE = {
    "H": (1, 1),  "He": (1, 18),
    "Li": (2, 1), "Be": (2, 2), "B": (2, 13), "C": (2, 14), "N": (2, 15), "O": (2, 16), "F": (2, 17), "Ne": (2, 18),
    "Na": (3, 1), "Mg": (3, 2), "Al": (3, 13), "Si": (3, 14), "P": (3, 15), "S": (3, 16), "Cl": (3, 17), "Ar": (3, 18),
    "K": (4, 1),  "Ca": (4, 2),
}
HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
    Chem.rdchem.HybridizationType.UNSPECIFIED: 5,
    Chem.rdchem.HybridizationType.OTHER: 5,
}


def atom_properties_to_tensor(mol):
    """
    RDKit分子から原子の周期, 族, 電荷, 混成軌道を取得し、PyTorchのテンソル形式に変換する。

    Parameters:
        mol (Chem.Mol): RDKit 分子オブジェクト

    Returns:
        torch.Tensor: 各原子の周期, 族, 電荷, 混成軌道の情報を格納したテンソル
    """
    data = []
    for atom in mol.GetAtoms():
        atomic_symbol = atom.GetSymbol()
        period, group = PERIODIC_TABLE[atomic_symbol]
        charge = atom.GetFormalCharge()
        hybridization = HYBRIDIZATION_MAP[atom.GetHybridization()] 
        
        data.append([period, group, charge, hybridization])
    
    # PyTorchテンソルに変換
    return torch.tensor(data, dtype=torch.int32)

# サンプル分子（エタノール）
mol = Chem.MolFromSmiles('C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]')

# テンソル化
tensor = atom_properties_to_tensor(mol)

# 結果表示
print("Tensor:\n", tensor)
