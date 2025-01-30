import os
import pandas as pd
import yaml

from lib.vocab import Vocab

def main(work_dir, vocab_file, spectra_df_file, output_dir, filter_setting_file, max_seq_len, overwrite):
    if os.path.exists(vocab_file):
        vocab = Vocab.load(vocab_file)
    else:
        raise FileNotFoundError(f'Vocab file not found: {vocab_file}')
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'ms_tensor.pt')
    failed_smiles_file = os.path.join(output_dir, 'error_ms.txt')
    tensor_stats_file = os.path.join(output_dir, 'tensor_stats_ms.txt')

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
    

    with open(filter_setting_file, 'r') as f:
        filter_setting = yaml.safe_load(f)
    vocab.set_ms_filter_config(filter_setting)

    spectra_df = pd.read_parquet(spectra_df_file)

    vocab.tensorize_msspectra(spectra_df, max_seq_len, output_file)

if __name__ == '__main__':
    import warnings
    import argparse
    warnings.simplefilter('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', "--work_dir", type = str, required=True)
    parser.add_argument('-v', "--vocab_file_name", type = str, default='')
    parser.add_argument('-i', '--spectra_df_file', type = str, required=True)
    parser.add_argument('-o', '--tensor_dir_name', type = str, required=True)
    parser.add_argument('-f', '--filter_setting_file', type = str, required=True)
    parser.add_argument('-seq', '--max_seq_len', type = int, required=True)
    parser.add_argument('-ow', '--overwrite', action='store_true')

    args = parser.parse_args()

    work_dir = args.work_dir
    vocab_file = os.path.join(work_dir, args.vocab_file_name)
    spectra_df_file = args.spectra_df_file
    output_dir = os.path.join(work_dir, args.tensor_dir_name)
    filter_setting_file = args.filter_setting_file
    max_seq_len = args.max_seq_len
    overwrite = args.overwrite
    
    main(
        work_dir=work_dir,
        vocab_file=vocab_file,
        spectra_df_file=spectra_df_file,
        output_dir=output_dir,
        filter_setting_file=filter_setting_file,
        max_seq_len=max_seq_len,
        overwrite=overwrite
    )
