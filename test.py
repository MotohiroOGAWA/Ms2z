from rdkit import Chem
# from rdkit.Chem import Draw
from tqdm import tqdm
import torch
import re

from model.utils import *
from model.fragment import Fragment
from model.fragment_bond import *
from model.vocab import Vocab



if __name__ == '__main__':
    code = 'mono_test'
    code = 'flow_test'
    if code == 'flow_test':
        monoatomic_tokens_file = '/workspaces/Ms2z/mnt/data/graph/pubchem/monoatomic_tokens.txt'
        fragment_counter_file = '/workspaces/Ms2z/mnt/data/graph/pubchem/fragment_counter.pkl'
        smiles_file = '/workspaces/Ms2z/mnt/data/smiles/pubchem/pubchem_smiles_10k.pkl'
        failed_smiles_file = '/workspaces/Ms2z/mnt/data/smiles/pubchem/failed_smiles_10k.txt'
        
        max_seq_len = 100
        vocab = Vocab(monoatomic_tokens_file, fragment_counter_file, threshold=0)

        smiles_list = read_smiles(smiles_file, binary=True)

        total_cnt = 0
        success_cnt = 0
        vocab_tensors = []
        order_tensors = []
        mask_tensors = []
        with tqdm(total=len(smiles_list), mininterval=0.5) as pbar:
            for smiles1 in smiles_list:
                for smiles in smiles1.split('.'):
                    try:
                    # if True:
                        output_smiles = 'None'
                        input_mol = Chem.MolFromSmiles(smiles)
                        input_smiles = Chem.MolToSmiles(input_mol, canonical=True)
                        # print(f"Input SMILES: {input_smiles}")
                        tensor = vocab.tensorize(input_mol, max_seq_len=max_seq_len)
                        if tensor is None:
                            # print(f'Input SMILES: {input_smiles}')
                            raise ValueError('Tensorization Failed')
                        
                        output_mol = vocab.detensorize(*tensor)
                        output_smiles = Chem.MolToSmiles(output_mol, canonical=True)
                        # print(f'detensorize: {output_smiles}')
                        result = input_smiles == output_smiles

                        if not result:
                            # print(f'Input SMILES: {input_smiles}')
                            # print(f'detensorize: {output_smiles}')
                            raise ValueError('Isomorphic Check Failed')

                        vocab_tensor, order_tensor, mask_tensor = tensor
                        vocab_tensors.append(vocab_tensor)
                        order_tensors.append(order_tensor)
                        mask_tensors.append(mask_tensor)

                        success_cnt += 1
                    except Exception as e:
                    # elif False:
                        if str(e) == 'Error: Ring not in vocabulary.':
                            total_cnt -= 1
                            continue
                        elif str(e).startswith('Not Found Atom Token'):
                            total_cnt -= 1
                            continue
                        with open(failed_smiles_file, 'a') as f:
                            f.write(f'{total_cnt}\t{smiles}\t{output_smiles}\t{str(e)}\n')
                            pass
                    finally:
                    # if True:
                        pbar.update(1)
                        total_cnt += 1
                        pbar.set_postfix_str(f'Success: {success_cnt}/{total_cnt} ({success_cnt/total_cnt:.2%})')
        vocab_tensors = torch.stack(vocab_tensors)
        order_tensors = torch.stack(order_tensors)
        mask_tensors = torch.stack(mask_tensors)
        tensor_file = '/workspaces/Ms2z/mnt/data/graph/pubchem/tensor/vocab_tensors.pt'

        torch.save({
            'vocab': vocab_tensors,
            'order': order_tensors,
            'mask': mask_tensors,
            'length': max_seq_len,
        }, tensor_file)
    elif code == 'mono_test':
        monoatomic_tokens_file = '/workspaces/Ms2z/mnt/data/graph/pubchem/monoatomic_tokens.txt'
        fragment_counter_file = '/workspaces/Ms2z/mnt/data/graph/pubchem/fragment_counter.pkl'
        
        max_seq_len = 100
        vocab = Vocab(monoatomic_tokens_file, fragment_counter_file, threshold=0)

        smiles_list = [
            # 'COc1ccc(C)cc1S(=O)(=O)C(C)C',
            'CCCCCCCCCC(CCCCCC)COC(=O)CCCCCCCCC(CCCCCCCCCC(=O)OCCC(CCCCCC)CCCCCCCC)N(CCC)C(=O)CCCCCCCC',
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

                    if not result:
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