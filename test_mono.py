from rdkit import Chem
# from rdkit.Chem import Draw
from tqdm import tqdm
import torch

from model.utils import *
from model.fragment import Fragment
from model.fragment_bond import *
from model.vocab import Vocab



if __name__ == '__main__':
    code = 'mono_test'
    # code = 'flow_test'
    if code == 'mono_test':
        monoatomic_tokens_file = '/workspaces/Ms2z/mnt/data/graph/pubchem/monoatomic_tokens.txt'
        fragment_counter_file = '/workspaces/Ms2z/mnt/data/graph/pubchem/fragment_counter.pkl'
        
        max_seq_len = 100
        vocab = Vocab(monoatomic_tokens_file, fragment_counter_file, threshold=0)

        smiles_list = [
            # 'COc1ccc(C)cc1S(=O)(=O)C(C)C',
            # 'O=P(O)(O)O',
            # 'Cc1nc(C)c2ncn(C3OC(CO)C(O)C3O)c2n1',
            # 'Cc1nc(C)c2ncn(C3OC(CO)C(O)C3O)c2n1',
            # 'CCCCCCC=CN(CCCCCCCC)CCCCCCCC',
            # 'O=C1c2ccc(NCCNCCO)c(NCCNCCO)c2C(=O)c2c(O)ccc(O)c21',
            # 'O=c1c2ccc(O)cc2c(=O)c2cc3c(=O)c4cc(O)ccc4c(=O)c3cc12',
            # 'NCC(C1=CC=CCC=C1)c1ccccc1',
            # 'CC(=CC=O)c1ccc(N(C)C)cc1',
            # 'C=CCCCC(O)C(O)CO',
            # 'COc1ccc(O)c(C(=O)Cl)c1',
            # 'Cc1cccc(C(=O)Nc2ccc(N(C)CCc3ccncc3)cc2)c1',
            # 'CCCC(C)(C)C(C)NC(=O)C(CCC(N)=O)NC(=O)C(C)(OC(C)C)OC(C)C',
            'COc1ccc(C(c2ccc(O)c(C)c2)C(c2ccc(O)c(C)c2)c2ccc(O)c(C)c2)cc1C',
            ]

        with tqdm(total=len(smiles_list), mininterval=0.5) as pbar:
            for smiles1 in smiles_list:
                for smiles in smiles1.split('.'):
                    output_smiles = 'None'
                    input_mol = Chem.MolFromSmiles(smiles)
                    input_smiles = Chem.MolToSmiles(input_mol, canonical=True)
                    print(f"Input SMILES: {input_smiles}")
                    tensor = vocab.tensorize(input_mol, max_seq_len=max_seq_len)
                    if tensor is None:
                        # print(f'Input SMILES: {input_smiles}')
                        raise ValueError('Tensorization Failed')
                    
                    output_mol = vocab.detensorize(*tensor)
                    output_smiles = Chem.MolToSmiles(output_mol, canonical=True)
                    # print(f'detensorize: {output_smiles}')
                    result = input_smiles == output_smiles

                    if result:
                        print(f'Isomorphic Check Passed')
                    else:
                        # print(f'Input SMILES: {input_smiles}')
                        # print(f'detensorize: {output_smiles}')
                        raise ValueError('Isomorphic Check Failed')
                    
    elif True:
        # 分子
        mol = Chem.MolFromSmiles("CCO[C]C")

        # 部分構造
        substructure = Chem.MolFromSmarts("CO")

        matches = mol.GetSubstructMatches(substructure)

        print("Matching atoms:", matches)