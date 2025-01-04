from rdkit import Chem
from rdkit.Chem import MACCSkeys
# from rdkit.Chem import Draw
from tqdm import tqdm
import torch
import re
import os

from lib.utils import *
from lib.fragment import Fragment
from lib.fragment_bond import *
from lib.vocab import Vocab


# print('\n'.join([f'{i}: {vocab}' for i, vocab in enumerate(vocab_list)]))
if __name__ == '__main__':
    vocab_file = '/workspaces/Ms2z/mnt/data/graph/pubchem_1M/vocab.pkl'

    monoatomic_tokens_file = '/workspaces/Ms2z/mnt/data/graph/pubchem_1M/monoatomic_tokens.txt'
    fragment_counter_file = '/workspaces/Ms2z/mnt/data/graph/pubchem_1M/fragment_counter.pkl'

    smiles_file = '/workspaces/Ms2z/mnt/data/smiles/pubchem/pubchem_smiles_10k.pkl'
    work_dir = '/workspaces/Ms2z/mnt/data/graph/pubchem_1k'
    failed_smiles_file = '/workspaces/Ms2z/mnt/data/smiles/pubchem/failed_smiles_10k.txt'

    # smiles_file = '/workspaces/Ms2z/mnt/data/smiles/pubchem/pubchem_smiles_1M.pkl'
    # work_dir = '/workspaces/Ms2z/mnt/data/graph/pubchem_1M'
    # failed_smiles_file = '/workspaces/Ms2z/mnt/data/smiles/pubchem/failed_smiles_1M.txt'
    
    threshold = 2
    max_seq_len = 100

    if os.path.exists(vocab_file):
        vocab = Vocab.load(vocab_file)
    else:
        vocab = Vocab(monoatomic_tokens_file, fragment_counter_file, threshold=threshold, save_path=vocab_file)

    smiles_list = read_smiles(smiles_file, binary=True)

    total_cnt = 0
    success_cnt = 0
    vocab_tensors = []
    order_tensors = []
    mask_tensors = []
    fp_tensors = []
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
                    maccs_fp = MACCSkeys.GenMACCSKeys(input_mol)
                    fp_tensor = torch.tensor(list(maccs_fp), dtype=torch.float64)

                    vocab_tensors.append(vocab_tensor)
                    order_tensors.append(order_tensor)
                    mask_tensors.append(mask_tensor)
                    fp_tensors.append(fp_tensor)

                    success_cnt += 1
                except Exception as e:
                # elif False:
                    # if str(e) == 'Error: Ring not in vocabulary.':
                    #     total_cnt -= 1
                    #     continue
                    # elif str(e).startswith('Not Found Atom Token'):
                    #     total_cnt -= 1
                    #     continue
                    with open(failed_smiles_file, 'a') as f:
                        error_message = str(e).replace('\n', ', ')
                        f.write(f'{total_cnt}\t{smiles}\t{output_smiles}\t{error_message}\n')
                        pass
                finally:
                # if True:
                    pbar.update(1)
                    total_cnt += 1
                    pbar.set_postfix_str(f'Success: {success_cnt}/{total_cnt} ({success_cnt/total_cnt:.2%})')
    vocab_tensors = torch.stack(vocab_tensors)
    order_tensors = torch.stack(order_tensors)
    mask_tensors = torch.stack(mask_tensors)
    fp_tensors = torch.stack(fp_tensors)
    tensor_file = os.path.join(work_dir, 'tensor', 'vocab_tensors.pt')
    os.makedirs(os.path.dirname(tensor_file), exist_ok=True)
    torch.save({
        'vocab': vocab_tensors,
        'order': order_tensors,
        'mask': mask_tensors,
        'length': max_seq_len,
        'vocab_size': len(vocab),
    }, tensor_file)
    torch.save(fp_tensors, os.path.join(work_dir, 'tensor', 'fp_tensors.pt'))

    max_valid_len_counts = torch.max(torch.sum(mask_tensors, dim=1))
    print(f'Maximum valid length: {max_valid_len_counts}')