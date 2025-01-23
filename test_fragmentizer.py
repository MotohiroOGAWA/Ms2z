from lib.fragmentizer import Fragmentizer

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

if __name__ == '__main__':
    smiles = 'CCCN=C1CC(CCc2cc(cc(CCC=COCCCO)c2O)-c2oc3cc(C(=O)OC)c(OC)c(O)c3c(=O)c2OC)CC(=C=C2CCCC(C2)C=C2CCCCC2)C1(C)CC'
    smiles = 'NCCOP(O)(O)=O'
    smiles = 'CCCCCC#CC#CC=CCC(C)CC1CC(C)C(=C2C(=O)C(C)N(C)C2=O)N1'
    # smiles = 'CC(=C=C2CCCCC2)C(C)CC'
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    # Run the check
    if not mol.GetNumConformers():
        AllChem.EmbedMolecule(mol)

    conformer = mol.GetConformer()
    
    # 原子1と原子2の座標を取得
    atom1_idx = 0
    atom2_idx = 1
    pos1 = np.array(conformer.GetAtomPosition(atom1_idx))
    pos2 = np.array(conformer.GetAtomPosition(atom2_idx))

    # 原子間の距離を計算
    distance = np.linalg.norm(pos1 - pos2)

    fragmentizer = Fragmentizer()
    motifs, _ = fragmentizer.split_to_motif(Chem.MolFromSmiles(smiles), max_attach_atom_cnt=2)
    print(motifs)
    