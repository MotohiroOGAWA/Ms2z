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

def main(work_dir, vocab_file, smiles_file, output_dir, max_seq_len, save_mode, overwrite, progress=True):
    if os.path.exists(vocab_file):
        vocab = Vocab.load(vocab_file, message=progress)
    else:
        raise FileNotFoundError(f'Vocab file not found: {vocab_file}')
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'tree_tensor.pt')
    output_smi_file = os.path.join(output_dir, 'smiles_data')
    failed_smiles_file = os.path.join(output_dir, 'error_smiles.txt')
    tensor_stats_file = os.path.join(output_dir, 'tensor_stats_chem.txt')

    # Check if output file exists when overwrite is False
    if os.path.exists(output_file) and not overwrite:
        user_input = input(f'Output file already exists: {output_file}. Do you want to overwrite it? (yes/no): ').strip().lower()
        if user_input != 'yes' and user_input != 'y':
            print('Execution aborted by user.')
            return
        
    elif os.path.exists(output_file) and overwrite:
        os.remove(output_file)
        if os.path.exists(failed_smiles_file):
            os.remove(failed_smiles_file)
        if os.path.exists(tensor_stats_file):
            os.remove(tensor_stats_file)

    smiles_list = read_smiles(smiles_file, duplicate=False)

    total_cnt = 0
    success_cnt = 0
    token_tensors = []
    order_tensors = []
    mask_tensors = []
    valid_smiles_list = []
    invalid_smiles_list = []
    
    iterator = tqdm(smiles_list, mininterval=0.5) if progress else smiles_list
    for smiles in iterator:
        try:
        # if True:
            output_smiles = 'None'
            input_mol = Chem.MolFromSmiles(smiles)
            input_smiles = Chem.MolToSmiles(input_mol, canonical=True)

            tensor = vocab.tensorize(input_mol, max_seq_len) # tensorize
            if tensor is None:
                raise ValueError('TensorizationError:')
            
            output_mol = vocab.detensorize(*tensor) # detensorize
            output_smiles = Chem.MolToSmiles(output_mol, canonical=True)
            result = input_smiles == output_smiles

            if not result:
                raise ValueError('IsomorphicError:')
            
            token_tensor, order_tensor, mask_tensor = tensor

            token_tensors.append(token_tensor)
            order_tensors.append(order_tensor)
            mask_tensors.append(mask_tensor)
            valid_smiles_list.append(input_smiles)

            success_cnt += 1

        except Exception as e:
        # elif False:
            error_message = str(e).replace('\n', ', ')
            invalid_smiles_list.append(f'{total_cnt+1}\t{smiles}\t{output_smiles}\t{error_message}\n')

            # with open(failed_smiles_file, 'a') as f:
            #     error_message = str(e).replace('\n', ', ')
            #     f.write(f'{total_cnt}\t{smiles}\t{output_smiles}\t{error_message}\n')
            #     pass
        finally:
        # if True:
            total_cnt += 1
            if progress:
                iterator.update(1)
                iterator.set_postfix_str(f'Success: {success_cnt}/{total_cnt} ({success_cnt/total_cnt:.2%})')

    if progress:
        iterator.close()

    token_tensors = torch.stack(token_tensors)
    order_tensors = torch.stack(order_tensors)
    mask_tensors = torch.stack(mask_tensors)

    max_valid_len_counts = torch.max(torch.sum(~mask_tensors, dim=1)).item()

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    torch.save({
        'token': token_tensors,
        'order': order_tensors,
        'mask': mask_tensors,
        'length': max_seq_len,
        'vocab_size': len(vocab),
        'attached_motif_len': vocab.attached_motif_len,
        'max_valid_seq_len': max_valid_len_counts,
    }, output_file)

    if 'b' in save_mode:
        dill.dump(valid_smiles_list, open(output_smi_file+'.pkl', 'wb'))

    if 't' in save_mode:
        with open(output_smi_file+'.txt', 'w') as f:
            f.writelines(smi+'\n' for smi in valid_smiles_list)

    if progress:
        print(f'Maximum valid length: {max_valid_len_counts}')

    with open(tensor_stats_file, 'w') as f:
        f.write(f'Size:\t{token_tensors.size(0)}\n')
        f.write(f'Total:\t{total_cnt}\n')
        f.write(f'Maximum valid length:\t{max_valid_len_counts}\n')

    if len(invalid_smiles_list) > 0:
        with open(failed_smiles_file, 'w') as f:
            f.writelines(invalid_smiles_list)

    print(f'Success: {success_cnt}/{total_cnt} ({success_cnt/total_cnt:.2%})')
    
    return success_cnt, total_cnt

# print('\n'.join([f'{i}: {vocab}' for i, vocab in enumerate(vocab_list)]))
if __name__ == '__main__':
    import warnings
    import argparse
    warnings.simplefilter('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', "--work_dir", type = str, required=True)
    parser.add_argument('-v', "--vocab_file_name", type = str, default='')
    parser.add_argument('-i', '--smiles_file', type = str, required=True)
    parser.add_argument('-o', '--tensor_dir_name', type = str, required=True)
    parser.add_argument('-seq_len', '--max_seq_len', type = int, required=True)
    parser.add_argument('-save_mode', '--save_mode', default='bt', type = str)
    parser.add_argument('-ow', '--overwrite', action='store_true')
    parser.add_argument('--no_progress', action='store_true')

    args = parser.parse_args()

    work_dir = args.work_dir
    vocab_file = os.path.join(work_dir, args.vocab_file_name)
    smiles_file = args.smiles_file
    output_dir = os.path.join(work_dir, args.tensor_dir_name)
    max_seq_len = args.max_seq_len
    
    main(
        work_dir=work_dir,
        vocab_file=vocab_file,
        smiles_file=smiles_file,
        output_dir=output_dir,
        max_seq_len=max_seq_len,
        save_mode=args.save_mode,
        overwrite=args.overwrite,
        progress=not args.no_progress
    )
