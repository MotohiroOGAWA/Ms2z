from model import *
from model.utils import *
from model.fragment import alt_atom
from rdkit import Chem
import copy




if __name__ == '__main__':
    # fragment_bond_list = FragBondList([(6, '-')])
    # fragment = Fragment('CCCCCCC', fragment_bond_list)
    # print(fragment)

    smiles = 'c1ccc2c(c1)Cc1cc3c(cc1C2)Cc1ccccc1C3'
    frag_bond_list = [(0, '-'), (6, '='), (13, '='), (14, '='), (18, '-'), (21, '=')]
    frag_bond_list = FragBondList(frag_bond_list)
    rwmol = Chem.RWMol(Chem.MolFromSmiles(smiles))

    fragment = Fragment(smiles, frag_bond_list)

    mol = copy.deepcopy(fragment.mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    smiles = Chem.MolToSmiles(mol)
    
    print(Chem.MolToSmiles(mol))
    print(fragment.bond_list)
