from rdkit import Chem
from rdkit.Chem import MACCSkeys
# from rdkit.Chem import Draw
from tqdm import tqdm
import torch
import re
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from lib.utils import *
from lib.fragment import Fragment
from lib.fragment_bond import *
from lib.vocab import Vocab


def process_smiles_batch(smiles_batch, vocab_file, max_seq_len):
    vocab = Vocab.load(vocab_file, message=False)
    """Process a batch of SMILES strings."""
    batch_results = {
        'length': max_seq_len,
        'vocab_size': len(vocab),
        'failed_smiles': [],
        'max_valid_seq_len': -1,
        'max_level': -1,
        'total_cnt': -1,
        'success_cnt': -1,
        'tensors_by_level': {
            -1: {
            'vocab': [],
            'order': [],
            'mask': [],
            'fingerprints': [],
            'fp_size': -1,
            'smiles': [],
            }
        }
    }

    by_level = True
    total_cnt = 0
    success_cnt = 0
    vocab_tensors = []
    order_tensors = []
    mask_tensors = []
    fp_tensors = []
    smiles_list = []

    vocab_tensors_by_level = defaultdict(list)
    order_tensors_by_level = defaultdict(list)
    mask_tensors_by_level = defaultdict(list)
    fp_tensors_by_level = defaultdict(list)
    smiles_list_by_level = defaultdict(list)
    tree_str_by_level = defaultdict(list)

    for smiles in smiles_batch:
        try:
            # Convert SMILES to RDKit Mol object
            output_smiles = 'None'
            input_mol = Chem.MolFromSmiles(smiles)
            input_smiles = Chem.MolToSmiles(input_mol, canonical=True)

            # Tensorize the molecule
            tensor = vocab.tensorize(input_mol, max_seq_len=max_seq_len)
            if tensor is None:
                raise ValueError('Tensorization Failed')

            # Detensorize and validate
            output_mol = vocab.detensorize(*tensor)
            output_smiles = Chem.MolToSmiles(output_mol, canonical=True)
            if input_smiles != output_smiles:
                raise ValueError('Isomorphic Check Failed')

            # Extract tensors
            vocab_tensor, order_tensor, mask_tensor = tensor
            maccs_fp = MACCSkeys.GenMACCSKeys(input_mol)
            fp_tensor = torch.tensor(list(maccs_fp), dtype=torch.float32)

            # Append results
            vocab_tensors.append(vocab_tensor)
            order_tensors.append(order_tensor)
            mask_tensors.append(mask_tensor)
            fp_tensors.append(fp_tensor)
            smiles_list.append(input_smiles)

            success_cnt += 1

            # Tensorize by level
            if by_level:
                tensor_by_level = vocab.tensorize_by_level(*tensor)
                for i, data in tensor_by_level.items():
                    try:
                        level_tensor = data['tensor']
                        level_vocab, level_order, level_mask = level_tensor
                        level = data['level']
                        output_mol = vocab.detensorize(*level_tensor)
                        output_smiles = Chem.MolToSmiles(output_mol, canonical=True)
                        maccs_fp_level = MACCSkeys.GenMACCSKeys(output_mol)
                        fp_tensor_level = torch.tensor(list(maccs_fp_level), dtype=torch.float32)

                        tree_str = output_smiles + ',' + ','.join([str(x.item()) for x in level_vocab[level_mask]])

                        if tree_str not in tree_str_by_level[level]:
                            vocab_tensors_by_level[level].append(level_vocab)
                            order_tensors_by_level[level].append(level_order)
                            mask_tensors_by_level[level].append(level_mask)
                            fp_tensors_by_level[level].append(fp_tensor_level)
                            smiles_list_by_level[level].append(output_smiles)
                            tree_str_by_level[level].append(tree_str)
                    except Exception as e:
                        pass

        except Exception as e:
            error_message = str(e).replace('\n', ', ')
            batch_results['failed_smiles'].append(f'{smiles}\t{output_smiles}\t{error_message}\n')

        finally:
            total_cnt += 1

    vocab_tensors = torch.stack(vocab_tensors)
    order_tensors = torch.stack(order_tensors)
    mask_tensors = torch.stack(mask_tensors)
    fp_tensors = torch.stack(fp_tensors)

    max_valid_len_counts = torch.max(torch.sum(mask_tensors, dim=1)).item()
    max_level = torch.max(order_tensors[:,:,4]).item()

    batch_results['max_valid_seq_len'] = max_valid_len_counts
    batch_results['max_level'] = max_level
    batch_results['total_cnt'] = total_cnt
    batch_results['success_cnt'] = success_cnt

    batch_results['tensors_by_level'][-1] = {
        'vocab': vocab_tensors,
        'order': order_tensors,
        'mask': mask_tensors,
        'fingerprints': fp_tensors,
        'fp_size': fp_tensors.size(1),
        'smiles': smiles_list,
        'tree_str': [],
    }

    for level in range(max_level+1):
        if len(vocab_tensors_by_level[level]) == 0:
            continue
        vocab_tensors = torch.stack(vocab_tensors_by_level[level])
        order_tensors = torch.stack(order_tensors_by_level[level])
        mask_tensors = torch.stack(mask_tensors_by_level[level])
        fp_tensors = torch.stack(fp_tensors_by_level[level])

        batch_results['tensors_by_level'][level] = {
            'vocab': vocab_tensors,
            'order': order_tensors,
            'mask': mask_tensors,
            'fingerprints': fp_tensors,
            'fp_size': fp_tensors.size(1),
            'smiles': smiles_list_by_level[level],
            'tree_str': tree_str_by_level[level],
        }

    return batch_results





def main(
        work_dir, vocab_file, monoatomic_tokens_file, smiles_counter_file, joint_counter_file, 
        smiles_file, output_file,
        threshold, max_seq_len,
        ncpu,
        batch_size,
        overwrite=False,
        ):
    if not overwrite and os.path.exists(vocab_file):
        vocab = Vocab.load(vocab_file)
    else:
        vocab = Vocab(monoatomic_tokens_file, smiles_counter_file, joint_counter_file, attachment_threshold=threshold, save_path=vocab_file)

    smiles_list = read_smiles(smiles_file, duplicate=False)
    smiles_batches = [smiles_list[i:i+batch_size] for i in range(0, len(smiles_list), batch_size)]

    by_level = True
    total_cnt = 0
    success_cnt = 0

    max_valid_len_counts = -1
    max_level = -1
    vocab_size = len(vocab)

    failed_smiles_file = os.path.join(work_dir, 'error_smiles.txt')
    with open(failed_smiles_file, 'w') as f:
        f.write('')

    # Parallel processing
    results = {}
    failed_smiles_list = []
    tree_str_by_level = defaultdict(list)

    with ProcessPoolExecutor(max_workers=ncpu) as executor:
        futures = [
            executor.submit(process_smiles_batch, batch, vocab_file, max_seq_len)
            for batch in smiles_batches
        ]
        pbar = tqdm(total=len(smiles_list), mininterval=0.5)
        for future in futures:
            batch_result = future.result()

            max_valid_len_counts = max(max_valid_len_counts, batch_result['max_valid_seq_len'])
            max_level = max(max_level, batch_result['max_level'])
            total_cnt += batch_result['total_cnt']
            success_cnt += batch_result['success_cnt']
            # Collect results by level
            for level, data in batch_result['tensors_by_level'].items():
                if level not in results:
                    results[level] = {
                        'vocab': [],
                        'order': [],
                        'mask': [],
                        'fp': [],
                        'smiles': [],
                    }
                
                tree_str_by_level[level].extend(data['tree_str'])
                results[level]['vocab'].append(data['vocab'])
                results[level]['order'].append(data['order'])
                results[level]['mask'].append(data['mask'])
                results[level]['fp'].append(data['fingerprints'])
                results[level]['smiles'].extend(data['smiles'])
            failed_smiles_list.extend(batch_result['failed_smiles'])
            pbar.update(batch_result['total_cnt'])
            pbar.set_postfix_str(f'Success: {success_cnt}/{total_cnt} ({success_cnt/total_cnt:.2%})')
            
    
    # Stack main tensors
    unique_tree_str_by_level = defaultdict(list)
    for level, data in tqdm(results.items(), desc='Stacking main tensors'):
        if level == -1:
            results[level]['vocab'] = torch.cat(data['vocab'], dim=0)
            results[level]['order'] = torch.cat(data['order'], dim=0)
            results[level]['mask'] = torch.cat(data['mask'], dim=0)
            results[level]['fp'] = torch.cat(data['fp'], dim=0)
        else:
            valid_indices = []
            for i, tree_str in enumerate(tree_str_by_level[level]):
                if tree_str not in unique_tree_str_by_level[level]:
                    unique_tree_str_by_level[level].append(tree_str)
                    valid_indices.append(i)
            valid_indices = torch.tensor(valid_indices, dtype=torch.long)
            results[level]['vocab'] = torch.cat(data['vocab'], dim=0)[valid_indices]
            results[level]['order'] = torch.cat(data['order'], dim=0)[valid_indices]
            results[level]['mask'] = torch.cat(data['mask'], dim=0)[valid_indices]
            results[level]['fp'] = torch.cat(data['fp'], dim=0)[valid_indices]
            results[level]['smiles'] = [data['smiles'][i] for i in valid_indices]

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tensor_stats_file = os.path.join(os.path.dirname(output_file), 'tensor_stats.txt')
    with open(tensor_stats_file, 'w') as f:
        f.write(f'level\tsize\n')
    
    for level in tqdm(range(-1,max_level+1)):
        if len(results[level]['vocab']) == 0:
            continue
        level_str = str(level).zfill(len(str(max_level)))
        vocab_tensors = results[level]['vocab']
        order_tensors = results[level]['order']
        mask_tensors = results[level]['mask']
        fp_tensors = results[level]['fp']
        smiles_list = results[level]['smiles']
        
        if level == -1:
            save_file = output_file
            smiles_file = os.path.join(os.path.dirname(output_file), 'smiles.txt')
        else:
            save_file = output_file.replace('.pt', f'_level{level_str}.pt')
            smiles_file = os.path.join(os.path.dirname(output_file), f'smiles_level{level_str}.txt')
        
        torch.save({
            'vocab': vocab_tensors,
            'order': order_tensors,
            'mask': mask_tensors,
            'length': max_seq_len,
            'vocab_size': vocab_size,
            'fingerprints': fp_tensors,
            'fp_size': fp_tensors.size(1),
            'max_valid_seq_len': max_valid_len_counts,
            'max_level': level,
        }, save_file)

        with open(smiles_file, 'w') as f:
            f.writelines([f'{smiles}\n' for smiles in smiles_list])

        with open(tensor_stats_file, 'a') as f:
            f.write(f'{level}\t{vocab_tensors.size(0)}\n')
    
    with open(tensor_stats_file, 'a') as f:
        f.write('\n')
        f.write(f'Length\tVocabSize\tTotalCnt\tSuccessCnt\tMaxValidSeqLen\tMaxLevel\n')
        f.write(f'{max_seq_len}\t{vocab_size}\t{total_cnt}\t{success_cnt}\t{max_valid_len_counts}\t{max_level}\n')

    with open(failed_smiles_file, 'w') as f:
        f.writelines(failed_smiles_list)


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
    ncpu = args.ncpu
    batch_size = args.batch_size
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
        ncpu=ncpu,
        batch_size=batch_size,
        overwrite=overwrite,
    )