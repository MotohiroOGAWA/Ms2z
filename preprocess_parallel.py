import subprocess
import os
import shutil
import dill
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import pandas as pd
import csv
from math import ceil
from lib.utils import read_smiles

script_path = "/workspaces/Ms2z/mnt/preprocess.py"

def main(args):
    temp_dir = os.path.join(args.work_dir, 'temp')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok = True)
    smile_file_paths = split_smiles_to_batches(args.smiles_file, args.batch_size, temp_dir)

    with ThreadPoolExecutor(max_workers=args.ncpu) as executor:
        futures = [executor.submit(run_in_subprocess, args) for args in generate_task_arguments(args, smile_file_paths)]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            future.result()
    
    attachment_df, motif_count_stats_df, valid_smiles_list, error_smiles_list\
          = merge_data_files(args, smile_file_paths)

    # motif_df.to_csv(
    #     os.path.join(args.work_dir, 'preprocess', 'motif_counter.tsv'), 
    #     sep='\t', 
    #     index=False, 
    #     quoting=csv.QUOTE_NONE,
    #     )
    os.makedirs(os.path.join(args.work_dir, 'preprocess'), exist_ok = True)
    with open(os.path.join(args.work_dir, 'preprocess', 'attachment_counter.tsv'), 'w') as f:
        for _, row in tqdm(attachment_df.iterrows(), desc="Writing attachment data", mininterval=0.1):
            f.write(row['motif'] + '\t' + row['output'] + '\n')

    print("Writing motif count stats data", end='...')
    os.makedirs(os.path.join(args.work_dir, 'preprocess', 'plot'), exist_ok = True)
    motif_count_stats_df.to_csv(
        os.path.join(args.work_dir, 'preprocess', 'plot', 'motif_count_stats.tsv'),
        sep='\t',
        index=False,
        columns=["BinStart", "Count", "CumulativeCount(Reversed)", "CumulativePercentage(Reversed)"],
    )
    print("done")

    print("Writing valid smiles data", end='...')
    if 'b' in  args.save_cnt_label:
        dill.dump(valid_smiles_list, open(os.path.join(args.work_dir, 'preprocess', 'valid_smiles.pkl'), "wb"))
    if 't' in  args.save_cnt_label:
        with open(os.path.join(args.work_dir, 'preprocess', 'valid_smiles.txt'), 'w') as f:
            f.writelines('\n'.join(valid_smiles_list))
    print("done")

    print("Writing error smiles data", end='...')
    with open(os.path.join(args.work_dir, 'preprocess', args.error_file), 'w') as f:
        f.writelines('\n'.join(error_smiles_list))
    print("done")

def split_smiles_to_batches(smiles_file, batch_size, temp_dir):
    os.makedirs(temp_dir, exist_ok = True)

    smiles_list = read_smiles(smiles_file)

    num_batches = ceil(len(smiles_list) / batch_size)
    zero_padding_width = len(str(num_batches))
    file_paths = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        folder_name = f"batch_{str(i // batch_size + 1).zfill(zero_padding_width)}"
        dir = os.path.join(temp_dir, folder_name)
        os.makedirs(dir, exist_ok = True)
        batch_file_path = os.path.join(dir, "smiles.txt")
        with open(batch_file_path, 'w') as batch_file:
            batch_file.writelines(b+'\n' for b in batch)
        file_paths.append(batch_file_path)
    return file_paths

def generate_task_arguments(args, smile_file_paths):
    for i, file_path in enumerate(smile_file_paths):
        yield {
            "-f": file_path, 
            "-o": os.path.dirname(file_path), 
            '--save_cnt_label': args.save_cnt_label,
            '--error_file': args.error_file,
            '--no_tqdm': True,
            }

def run_in_subprocess(args):
    command = [
        "python", script_path, 
        ]
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                command.append(key)
        else:
            command.extend([key, str(value)])
    subprocess.run(command)

def merge_data_files(args, file_paths):
    attachment_df = pd.DataFrame({}, columns=['motif', 'attachment', 'count'])
    motif_count_stats_df = pd.DataFrame({
        "BinStart": pd.Series(dtype="int32"),  
        "Count": pd.Series(dtype="int32"),    
        "CumulativeCount(Reversed)": pd.Series(dtype="int32"),  
        "CumulativePercentage(Reversed)": pd.Series(dtype="float64") 
    })
    valid_smiles_list = []
    error_smiles_list = []

    for file_path in tqdm(file_paths, desc="Merging data"):
        update_attachment_data  = []
        dirname = os.path.join(os.path.dirname(file_path), 'preprocess')

        # merge attachment counter data
        with open(os.path.join(dirname, 'attachment_counter.tsv'), 'r') as f:
            for line in f:
                motif, count, *attachments = line.split('\t')
                update_attachment_data .append({'motif': motif, 'attachment': '', 'count': int(count)})
                for attachment, atm_cnt in zip(attachments[::2], attachments[1::2]):
                    update_attachment_data .append({'motif': motif, 'attachment': attachment, 'count': int(atm_cnt)})
        update_attachment_df = pd.DataFrame(update_attachment_data, columns=['motif', 'attachment', 'count'])
        attachment_df = pd.concat([attachment_df, update_attachment_df])
        attachment_df = attachment_df.groupby(['motif', 'attachment'], as_index=False)['count'].sum()
        
        # merge motif count stats data
        update_motif_count_stats_df = pd.read_csv(os.path.join(dirname, 'plot', 'motif_count_stats.tsv'), sep='\t')
        motif_count_stats_df = pd.concat([motif_count_stats_df, update_motif_count_stats_df])
        motif_count_stats_df = motif_count_stats_df.groupby('BinStart', as_index=False).sum()

        # merge valid smiles data
        if os.path.exists(os.path.join(dirname, 'valid_smiles.pkl')):
            update_valid_smiles_list = dill.load(open(os.path.join(dirname, 'valid_smiles.pkl'), 'rb'))
        else:
            with open(os.path.join(dirname, 'valid_smiles.txt'), 'r') as f:
                update_valid_smiles_list = f.readlines()
        valid_smiles_list.extend(update_valid_smiles_list)

        # merge error smiles data
        with open(os.path.join(dirname, args.error_file), 'r') as f:
            update_error_smiles_list = f.readlines()
        error_smiles_list.extend(update_error_smiles_list)

    attachment_df = attachment_df.groupby("motif").apply(
        lambda group: f"{group[group['attachment'] == '']['count'].sum()}\t" + "\t".join(
            f"{attachment}\t{count}" for attachment, count in zip(group['attachment'], group['count']) if attachment != ''
        )
    ).reset_index(name="output")
    attachment_df['output'] = attachment_df['output'].str.replace(r'[\n]', ' ', regex=True)

    # calculate CumulativePercentage(Reversed) of motif count stats
    max_cumulative_count = motif_count_stats_df['CumulativeCount(Reversed)'].max()
    motif_count_stats_df['CumulativePercentage(Reversed)'] = motif_count_stats_df['CumulativeCount(Reversed)'] / max_cumulative_count * 100

    return attachment_df, motif_count_stats_df, valid_smiles_list, error_smiles_list

if __name__ == "__main__":
    # kill command
    # pkill -9 -f python
    # python /workspaces/Ms2z/mnt/preprocess_parallel.py -w /workspaces/Ms2z/mnt/data/graph/msfinder -i /workspaces/Ms2z/mnt/data/smiles/msfinder/filtered_smiles_mini.txt -ncpu 30 -batch 100 --save_cnt_label bt --error_file error_smiles.txt
    # python /workspaces/Ms2z/mnt/preprocess_parallel.py -w /workspaces/Ms2z/mnt/data/graph/test_pubchem_1M -i /workspaces/Ms2z/mnt/data/smiles/pubchem/pubchem_smiles_1M.txt -ncpu 50 -batch 10000 --save_cnt_label bt --error_file error_smiles.txt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', "--work_dir", type = str, required=True, help = "Path to the working directory")
    parser.add_argument('-i', "--smiles_file", type = str, required=True, help = "Path to the SMILES file")
    parser.add_argument('-ncpu', "--ncpu", type = int, default = 1, help = "Number of CPUs to use")
    parser.add_argument('-batch', '--batch_size', type = int, default = 1000, help = "Batch size for processing")
    parser.add_argument('--save_cnt_label', type = str, choices=['b', 't', 'bt', 'tb'], default = 'bt', help = "Type of count label to save")
    # parser.add_argument('--save_mol', action='store_true', help = "Save the created molecule data with RDKit")
    parser.add_argument('--error_file', type = str, default = 'error_data.txt', help = "Path to save data that caused errors")
    
    args = parser.parse_args()
    main(args)

