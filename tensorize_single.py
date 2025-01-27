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

def main(
        work_dir, vocab_file, monoatomic_tokens_file, smiles_counter_file, joint_counter_file, 
        smiles_file, output_file,
        threshold, max_seq_len,
        overwrite=False
        ):
    if not overwrite and os.path.exists(vocab_file):
        vocab = Vocab.load(vocab_file)
    else:
        vocab = Vocab(monoatomic_tokens_file, smiles_counter_file, joint_counter_file, threshold=threshold, save_path=vocab_file)

    smiles_list = read_smiles(smiles_file, duplicate=False)
    # smiles_list = [s for smiles in smiles_list for s in smiles.split('.')]

    by_level = True
    total_cnt = 0
    success_cnt = 0
    vocab_tensors = []
    order_tensors = []
    mask_tensors = []
    fp_tensors = []

    vocab_tensors_by_level = defaultdict(list)
    order_tensors_by_level = defaultdict(list)
    mask_tensors_by_level = defaultdict(list)
    fp_tensors_by_level = defaultdict(list)
    tree_str_by_level = defaultdict(list)

    failed_smiles_file = os.path.join(work_dir, 'error_smiles.txt')
    with open(failed_smiles_file, 'w') as f:
        f.write('')
        
    with tqdm(total=len(smiles_list), mininterval=0.5) as pbar:
        for smiles in smiles_list:
            # try:
            if True:
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
                fp_tensor = torch.tensor(list(maccs_fp), dtype=torch.float32)

                vocab_tensors.append(vocab_tensor)
                order_tensors.append(order_tensor)
                mask_tensors.append(mask_tensor)
                fp_tensors.append(fp_tensor)

                success_cnt += 1


                if by_level:
                    tensor_by_level = vocab.tensorize_by_level(*tensor)
                    
                    for i, data in tensor_by_level.items():
                        try:
                            tensor2 = data['tensor']
                            level = data['level']
                            vocab_tensor2, order_tensor2, mask_tensor2 = tensor2
                            output_mol = vocab.detensorize(*tensor2)
                            output_smiles = Chem.MolToSmiles(output_mol, canonical=True)
                            maccs_fp = MACCSkeys.GenMACCSKeys(output_mol)
                            fp_tensor = torch.tensor(list(maccs_fp), dtype=torch.float32)

                            tree_str = output_smiles + ',' + ','.join([str(x.item()) for x in vocab_tensor2[mask_tensor2]])

                            if tree_str not in tree_str_by_level[level]:
                                vocab_tensors_by_level[level].append(vocab_tensor2)
                                order_tensors_by_level[level].append(order_tensor2)
                                mask_tensors_by_level[level].append(mask_tensor2)
                                fp_tensors_by_level[level].append(fp_tensor)
                                tree_str_by_level[level].append(tree_str)

                        except Exception as e:
                            pass

            # except Exception as e:
            elif False:
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
            # finally:
            if True:
                pbar.update(1)
                total_cnt += 1
                pbar.set_postfix_str(f'Success: {success_cnt}/{total_cnt} ({success_cnt/total_cnt:.2%})')


    vocab_tensors = torch.stack(vocab_tensors)
    order_tensors = torch.stack(order_tensors)
    mask_tensors = torch.stack(mask_tensors)
    fp_tensors = torch.stack(fp_tensors)

    max_valid_len_counts = torch.max(torch.sum(mask_tensors, dim=1)).item()
    max_level = torch.max(order_tensors[:,:,4]).item()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save({
        'vocab': vocab_tensors,
        'order': order_tensors,
        'mask': mask_tensors,
        'length': max_seq_len,
        'vocab_size': len(vocab),
        'fingerprints': fp_tensors,
        'fp_size': fp_tensors.size(1),
        'max_valid_seq_len': max_valid_len_counts,
        'max_level': max_level,
    }, output_file)
    # torch.save(fp_tensors, os.path.join(work_dir, 'tensor', 'fp_tensors.pt'))

    print(f'Maximum valid length: {max_valid_len_counts}')
    print(f'Maximum level: {max_level}')


    tensor_stats_file = os.path.join(os.path.dirname(output_file), 'tensor_stats.txt')
    with open(tensor_stats_file, 'w') as f:
        f.write(f'level\tsize\n')
        f.write(f'-1\t{vocab_tensors.size(0)}\n')

    for level in tqdm(range(max_level+1)):
        if len(vocab_tensors_by_level[level]) == 0:
            continue
        level_str = str(level).zfill(len(str(max_level)))
        vocab_tensors = torch.stack(vocab_tensors_by_level[level])
        order_tensors = torch.stack(order_tensors_by_level[level])
        mask_tensors = torch.stack(mask_tensors_by_level[level])
        fp_tensors = torch.stack(fp_tensors_by_level[level])

        torch.save({
            'vocab': vocab_tensors,
            'order': order_tensors,
            'mask': mask_tensors,
            'length': max_seq_len,
            'vocab_size': len(vocab),
            'fingerprints': fp_tensors,
            'fp_size': fp_tensors.size(1),
            'max_valid_seq_len': max_valid_len_counts,
            'max_level': level,
        }, output_file.replace('.pt', f'_level{level_str}.pt'))

        with open(tensor_stats_file, 'a') as f:
            f.write(f'{level}\t{vocab_tensors.size(0)}\n')

# print('\n'.join([f'{i}: {vocab}' for i, vocab in enumerate(vocab_list)]))
if __name__ == '__main__':
    import warnings
    import argparse
    warnings.simplefilter('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', "--work_dir", type = str, required=True)
    parser.add_argument('-v', "--vocab_file_name", type = str, default='')
    parser.add_argument('-mt', "--monoatomic_tokens_file_name", type = str, default='')
    parser.add_argument('-sc', '--smiles_counter_file_name', type = str, default='')
    parser.add_argument('-jc', '--joint_counter_file_name', type = str, default='')
    parser.add_argument('-i', '--smiles_file_name', type = str, required=True)
    parser.add_argument('-o', '--tensor_file_name', type = str, required=True)
    parser.add_argument('-thres', '--threshold', type = int, required=True)
    parser.add_argument('-seq_len', '--max_seq_len', type = int, required=True)
    parser.add_argument('-ncpu', '--ncpu', type = int, required=True)
    parser.add_argument('-batch_size', '--batch_size', type = int, required=True)
    parser.add_argument('-ow', '--overwrite', action='store_true')

    args = parser.parse_args()

    work_dir = args.work_dir
    vocab_file = os.path.join(work_dir, args.vocab_file_name)
    monoatomic_tokens_file = os.path.join(work_dir, 'preprocess', args.monoatomic_tokens_file_name)
    smiles_counter_file = os.path.join(work_dir, 'preprocess', args.smiles_counter_file_name)
    joint_counter_file = os.path.join(work_dir, 'preprocess', args.joint_counter_file_name)
    smiles_file = os.path.join(work_dir, args.smiles_file_name)
    output_file = os.path.join(work_dir, 'tensor', args.tensor_file_name)
    threshold = args.threshold
    max_seq_len = args.max_seq_len
    overwrite = args.overwrite
    main(
        work_dir=work_dir,
        vocab_file=vocab_file,
        monoatomic_tokens_file=monoatomic_tokens_file,
        smiles_counter_file=smiles_counter_file,
        joint_counter_file=joint_counter_file,
        smiles_file=smiles_file,
        output_file=output_file,
        threshold=threshold,
        max_seq_len=max_seq_len,
        overwrite=overwrite,
    )
