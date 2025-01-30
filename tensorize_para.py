import os
import shutil
import dill
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from math import ceil
import torch

from lib.utils import read_smiles

script_path = "/workspaces/Ms2z/mnt/tensorize.py"

def main(args):
    output_dir = os.path.join(args.work_dir, args.tensor_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'tree_tensor.pt')
    failed_smiles_file = os.path.join(output_dir, 'error_smiles.txt')
    tensor_stats_file = os.path.join(output_dir, 'tensor_stats_chem.txt')

    # Check if output file exists when overwrite is False
    if os.path.exists(output_file) and not args.overwrite:
        user_input = input(f'Output file already exists: {output_file}. Do you want to overwrite it? (yes/no): ').strip().lower()
        if user_input != 'yes' and user_input != 'y':
            print('Execution aborted by user.')
            return
        
    elif os.path.exists(output_file) and args.overwrite:
        os.remove(output_file)
        if os.path.exists(failed_smiles_file):
            os.remove(failed_smiles_file)
        if os.path.exists(tensor_stats_file):
            os.remove(tensor_stats_file)
    # Create a temporary directory for batch processing
    temp_dir = os.path.join(args.work_dir, 'temp')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Split SMILES into batches and save to temporary files
    batch_file_paths = split_smiles_to_batches(args.smiles_file, args.batch_size, temp_dir)

    # Process each batch in parallel using subprocess
    success_cnt = 0
    total_cnt = 0
    with ThreadPoolExecutor(max_workers=args.ncpu) as executor:
        futures = [executor.submit(run_in_subprocess, args) for args in generate_task_arguments(args, batch_file_paths)]
        iterator = tqdm(futures, total=len(futures), desc="Processing batches")

        for future in iterator:
            _success_cnt, _total_cnt = future.result()
            success_cnt += _success_cnt
            total_cnt += _total_cnt
            iterator.set_postfix_str(f'Success: {success_cnt}/{total_cnt} ({success_cnt/total_cnt:.2%})')

    # Merge results from all batches
    merge_results(args, batch_file_paths)

    # Remove the temporary directory after processing
    shutil.rmtree(temp_dir)
    print("Tensorization complete!")

def split_smiles_to_batches(smiles_file, batch_size, temp_dir):
    """
    Splits the SMILES file into smaller batches and saves them as temporary files.

    Args:
        smiles_file (str): Path to the input SMILES file.
        batch_size (int): Number of SMILES per batch.
        temp_dir (str): Directory to store temporary batch files.

    Returns:
        list: A list of file paths for the batch files.
    """
    os.makedirs(temp_dir, exist_ok = True)

    smiles_list = read_smiles(smiles_file)

    num_batches = ceil(len(smiles_list) / batch_size)
    zero_padding_width = len(str(num_batches))
    batch_files = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        folder_name = f"batch_{str(i // batch_size + 1).zfill(zero_padding_width)}"
        dir = os.path.join(temp_dir, folder_name)
        os.makedirs(dir, exist_ok = True)
        batch_file_path = os.path.join(dir, "smiles.txt")
        with open(batch_file_path, 'w') as batch_file:
            batch_file.writelines(b+'\n' for b in batch)
        batch_files.append(batch_file_path)
    return batch_files


def generate_task_arguments(args, batch_files):
    """
    Generates the command-line arguments for the subprocess.

    Args:
        args (argparse.Namespace): Parsed arguments from the main script.
        batch_files (str): List of batch file paths.

    Returns:
        list: List of command-line arguments for the subprocess.
    """
    for batch_file in batch_files:
        yield [
            "-w", os.path.dirname(batch_file),
            "-v", os.path.join('../..', args.vocab_file),
            "-i", batch_file,
            "-o", args.tensor_dir_name,
            "-seq_len", str(args.max_seq_len),
            "-save_mode", "b",
            "-ow",
            "--no_progress",
        ]

def run_in_subprocess(args):
    command = [
        "python", script_path, 
        ]
    if isinstance(args, list):
        command.extend(args)
    elif isinstance(args, dict):
        for key, value in args.items():
            if isinstance(value, bool):
                if value:
                    command.append(key)
            else:
                command.extend([key, str(value)])

    try:
        # Run the subprocess and capture its output
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Extract `Success: X/Y (Z%)` from the output using regex
        match = re.search(r"Success:\s*(\d+)/(\d+)", result.stdout)
        if match:
            success_cnt = int(match.group(1))  # Extract X = number of successful cases
            total_cnt = int(match.group(2))    # Extract Y = total cases
            return success_cnt, total_cnt
        else:
            print("Could not find success count in output.")
            return 0, 0  # Return default values if parsing fails

    except subprocess.CalledProcessError as e:
        print(f"Error in subprocess: {e}")
        print(f"Command output: {e.output}")
        return 0, 0

def merge_results(args, batch_file_paths):
    """
    Merges tensor files and SMILES files from all batches into a single output.

    Args:
        args (argparse.Namespace): Parsed arguments from the main script.
        smile_file_paths (list): List of batch file paths.
    """
    output_dir = os.path.join(args.work_dir, args.tensor_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    tensor_data = {
        'token': [],
        'order': [],
        'mask': [],
        'length': 0,
        'vocab_size': 0,
        'attached_motif_len': 0,
        'max_valid_seq_len': 0,
    }
    smiles_data = []
    stats_data = {
        'Size': 0,
        'Total': 0,
        'Maximum valid length': 0,
    }
    error_smiles_list = []

    for i, batch_file in tqdm(enumerate(batch_file_paths), total=len(batch_file_paths), desc="Merging results"):
        batch_output_dir = os.path.dirname(batch_file)
        tensor_file = os.path.join(batch_output_dir, args.tensor_dir_name, "tree_tensor.pt")
        smiles_file = os.path.join(batch_output_dir, args.tensor_dir_name, "smiles_data.pkl")
        stats_file = os.path.join(batch_output_dir, args.tensor_dir_name, "tensor_stats_chem.txt")
        error_smiles_file = os.path.join(batch_output_dir, args.tensor_dir_name, "error_smiles.txt")
        start_idx = i * args.batch_size

        if os.path.exists(tensor_file):
            tensor = torch.load(tensor_file)
            tensor_data['token'].extend(tensor['token'])
            tensor_data['order'].extend(tensor['order'])
            tensor_data['mask'].extend(tensor['mask'])
            tensor_data['length'] = tensor['length']
            tensor_data['vocab_size'] = tensor['vocab_size']
            tensor_data['attached_motif_len'] = tensor['attached_motif_len']
            tensor_data['max_valid_seq_len'] = max(tensor_data['max_valid_seq_len'], tensor['max_valid_seq_len'])

        else:
            raise FileNotFoundError(f"Tensor file not found: {tensor_file}")
        
        if os.path.exists(smiles_file):
            smiles_data.extend(dill.load(open(smiles_file, 'rb')))
        else:
            raise FileNotFoundError(f"SMILES file not found: {smiles_file}")
        
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = {line.split(':\t')[0]: int(line.split(':\t')[1].strip()) for line in f.readlines()}
                stats_data['Size'] += stats['Size']
                stats_data['Total'] += stats['Total']
                stats_data['Maximum valid length'] = max(stats_data['Maximum valid length'], stats['Maximum valid length'])
        else:
            raise FileNotFoundError(f"Stats file not found: {stats_file}")

        if os.path.exists(error_smiles_file):
            with open(error_smiles_file, 'r') as f:
                error_smiles_list.extend("\t".join([str(int(s[0]) + start_idx)] + s[1:]) for s in (t.split("\t") for t in f.readlines()))

        else:
            raise FileNotFoundError(f"Error SMILES file not found: {error_smiles_file}")

    tensor_data['token'] = torch.stack(tensor_data['token'])
    tensor_data['order'] = torch.stack(tensor_data['order'])
    tensor_data['mask'] = torch.stack(tensor_data['mask'])
    
    assert stats_data['Size'] == len(tensor_data['token']) == len(tensor_data['order']) == len(tensor_data['mask']), "Mismatch between tensor data"
    assert stats_data['Size'] == len(smiles_data), "Mismatch between tensor and SMILES data"

    # Save the merged results
    final_tensor_file = os.path.join(output_dir, "tree_tensor.pt")
    final_smiles_file = os.path.join(output_dir, "smiles_data")
    final_stats_file = os.path.join(output_dir, "tensor_stats_chem.txt")
    final_error_file = os.path.join(output_dir, "error_smiles.txt")

    torch.save(tensor_data, final_tensor_file)

    if 'b' in args.save_mode:
        dill.dump(smiles_data, open(final_smiles_file+'.pkl', 'wb'))
    if 't' in args.save_mode:
        with open(final_smiles_file+'.txt', "w") as f:
            f.writelines([f"{smi}\n" for smi in smiles_data])

    with open(final_stats_file, 'w') as f:
        f.write(f'Size:\t{stats_data["Size"]}\n')
        f.write(f'Total:\t{stats_data["Total"]}\n')
        f.write(f'Maximum valid length:\t{stats_data["Maximum valid length"]}\n')
    
    if len(error_smiles_list) > 0:
        with open(final_error_file, 'w') as f:
            f.writelines(error_smiles_list)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--work_dir", type=str, required=True, help="Working directory")
    parser.add_argument("-v", "--vocab_file", type=str, required=True, help="Vocabulary file")
    parser.add_argument("-i", "--smiles_file", type=str, required=True, help="SMILES input file")
    parser.add_argument("-o", "--tensor_dir_name", type=str, required=True, help="Output directory for tensor files")
    parser.add_argument("-seq_len", "--max_seq_len", type=int, required=True, help="Maximum sequence length")
    parser.add_argument("-save_mode", "--save_mode", default='bt', type=str, help="Save mode for SMILES data")
    parser.add_argument("-batch", "--batch_size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("-ncpu", "--ncpu", type=int, default=1, help="Number of CPU cores to use for parallel processing")
    parser.add_argument("-ow", "--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    main(args)
