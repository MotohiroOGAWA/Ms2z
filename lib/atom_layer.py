from rdkit import Chem
from rdkit.Chem import rdchem
import torch

PERIODIC_TABLE = {
    # "H": (1, 1),  "He": (1, 18),
    # "Li": (2, 1), "Be": (2, 2), "B": (2, 13), "C": (2, 14), "N": (2, 15), "O": (2, 16), "F": (2, 17), "Ne": (2, 18),
    # "Na": (3, 1), "Mg": (3, 2), "Al": (3, 13), "Si": (3, 14), "P": (3, 15), "S": (3, 16), "Cl": (3, 17), "Ar": (3, 18),
    # "K": (4, 1),  "Ca": (4, 2),
    "C": (2, 14), "N": (2, 15), "O": (2, 16), "F": (2, 17),
    "Si": (3, 14), "P": (3, 15), "S": (3, 16), "Cl": (3, 17),
    "Br": (4, 17), "I": (5, 17),
}
unique_periods = sorted(set(value[0] for value in PERIODIC_TABLE.values()))
period_map = {period: idx for idx, period in enumerate(unique_periods)}
unique_groups = sorted(set(value[1] for value in PERIODIC_TABLE.values()))
group_map = {group: idx for idx, group in enumerate(unique_groups)}

HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
    Chem.rdchem.HybridizationType.UNSPECIFIED: 5,
    Chem.rdchem.HybridizationType.OTHER: 5,
}
unique_hybridizations = sorted(set(HYBRIDIZATION_MAP.values()))
hybridization_map = {hybridization: idx for idx, hybridization in enumerate(unique_hybridizations)}

BOND_TYPE_MAP = {
    rdchem.BondType.SINGLE: 0,
    rdchem.BondType.DOUBLE: 1,
    rdchem.BondType.TRIPLE: 2,
    rdchem.BondType.AROMATIC: 3,
}
unique_bond_types = sorted(set(BOND_TYPE_MAP.values()))
bond_type_map = {bond_type: idx for idx, bond_type in enumerate(unique_bond_types)}


def initialize_feature_validity():
    """
    特徴量の有効性を示すテンソルを初期化し、各特徴量に対応する範囲をTrueとする。

    Returns:
        dict: 各特徴量の有効領域を示すテンソルの辞書
    """
    total_feature_size = len(period_map) + len(group_map) + 1 + len(hybridization_map)
    validity_template = torch.zeros(total_feature_size, dtype=torch.bool)

    # Period有効性
    period_validity = validity_template.clone()
    period_validity[:len(period_map)] = True

    # Group有効性
    group_validity = validity_template.clone()
    group_validity[len(period_map):len(period_map) + len(group_map)] = True

    # Charge有効性
    charge_validity = validity_template.clone()
    charge_validity[len(period_map) + len(group_map)] = True

    # Hybridization有効性
    hybridization_validity = validity_template.clone()
    hybridization_start = len(period_map) + len(group_map) + 1
    hybridization_validity[hybridization_start:hybridization_start + len(hybridization_map)] = True

    return period_validity, group_validity, charge_validity, hybridization_validity

period_validity, group_validity, charge_validity, hybridization_validity = initialize_feature_validity()

def one_hot_vector(value, length):
    """
    指定されたインデックスを1にするone-hotベクトルを生成。
    """
    vector = [0] * length
    vector[value] = 1
    return vector

def atom_bond_properties_to_tensor(mol):
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

        # One-hotベクトルを生成
        period_vector = one_hot_vector(period_map[period], len(period_map))
        group_vector = one_hot_vector(group_map[group], len(group_map))
        charge_vector = [charge]  # 電荷はそのままスカラー値
        hybridization_vector = one_hot_vector(hybridization_map[hybridization], len(hybridization_map))
        
        atom_vector = period_vector + group_vector + charge_vector + hybridization_vector
        data.append(atom_vector)
    nodes = torch.tensor(data, dtype=torch.float32)
    
    data_edge = []
    data_attr = []
    # ring_info = mol.GetRingInfo()
    atr_arr = [0.0] * len(bond_type_map)

    
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = BOND_TYPE_MAP[bond.GetBondType()]
        bond_type_num = bond_type_map[bond_type]
        _attr_arr = atr_arr.copy()
        _attr_arr[bond_type_num] = 1.0


        # # 結合が属する環のインデックスを取得
        # bond_rings = [
        #     ring for ring in ring_info.BondRings() if bond.GetIdx() in ring
        # ]
        # num_rings = len(bond_rings)  # 結合が属するユニークな環の数
        
        data_edge.append([start, end])
        data_attr.append(_attr_arr)
    
    # PyTorchテンソルに変換
    edges = torch.tensor(data_edge, dtype=torch.int64).T # (2, num_edges)
    attr = torch.tensor(data_attr, dtype=torch.float32) # (num_edges, edge_dim)
    
    # PyTorchテンソルに変換
    return nodes, edges, attr


if __name__ == '__main__':
    # サンプル分子（エタノール）
    mol = Chem.MolFromSmiles('C1=CC=C2C=CC=CC2=C1')

    # テンソル化
    atom_tensor = atom_properties_to_tensor(mol)
    edge_tensor, edge_atr_tensor = bond_properties_to_tensor(mol)

    # 結果表示
    print("Atom Tensor:\n", atom_tensor)
    print("Edge Tensor:\n", edge_tensor)
    print("Edge Attribute Tensor:\n", edge_atr_tensor)
