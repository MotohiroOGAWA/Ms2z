from rdkit import Chem
from rdkit.Chem import MACCSkeys
import torch
import copy
import re




if __name__ == '__main__':
    # fragment_bond_list = FragBondList([(6, '-')])
    # fragment = Fragment('CCCCCCC', fragment_bond_list)
    # print(fragment)

    smiles = 'CC'
    mol = Chem.MolFromSmiles(smiles)

    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    fp_tensor = torch.tensor(list(maccs_fp), dtype=torch.int32)
    print(fp_tensor==1)

