from model.fragment import Fragment, FragBondList
from model.utils import *
from model.fragment import alt_atom
from rdkit import Chem
import copy
import re




if __name__ == '__main__':
    # fragment_bond_list = FragBondList([(6, '-')])
    # fragment = Fragment('CCCCCCC', fragment_bond_list)
    # print(fragment)

    smiles = 'C=C(C)C(C)(C)C(O)CCCCCCCCC(C)=O'
    frag_bond_list = [(0, '-'), (2, '='), (4, '#'), (6, '-'), (7, '-')] 
    frag_bond_list = FragBondList(frag_bond_list)
    fragment = Fragment(smiles, frag_bond_list)


    qry_smiles = 'CO'
    qry_frag_bond_list = [(0, '-'), (0, '-'), (1, '-')]
    # qry_smiles = 'OC'
    # qry_frag_bond_list = [(1, '-'), (1, '-'), (0, '-')]
    qry_frag_bond_list = FragBondList(qry_frag_bond_list)
    qry_fragment = Fragment(qry_smiles, qry_frag_bond_list)
    
    matches = fragment.GetSubstructMatches(qry_fragment)
    print(matches)
    print(qry_fragment)




    smarts = Chem.MolToSmarts(fragment.mol)
    pattern = r"\[#(\d+)\]" 
    print([int(match.group(1)) for match in re.finditer(pattern, smarts)])
    mol = copy.deepcopy(fragment.mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    smiles = Chem.MolToSmiles(mol)
    
    print(Chem.MolToSmiles(mol))
    print(fragment.bond_list)

    qry_smiles = 'CO'
    frag_bond_list = [(0, '-'), (0, '-')]
    qry_fragment = Fragment(qry_smiles, frag_bond_list)

