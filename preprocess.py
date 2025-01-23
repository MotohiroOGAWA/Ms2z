from lib.utils import *
from lib.fragmentizer import Fragmentizer
from lib.fragment import Fragment
from lib.motif import *
from stats.plot_utils import plot_counter_distribution

from collections import Counter, defaultdict
from tqdm import tqdm
import os

from rdkit import Chem

# Disable RDKit logging
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)  # Only show critical errors, suppress warnings and other messages


def main(args):
    os.makedirs(args.save_dir, exist_ok = True)

    suppl = None
    mols = []
    if args.file_path.endswith('.sdf'):
        if not args.no_tqdm:
            print("Reading Chem.Mol from SDF file")
        suppl = Chem.SDMolSupplier(args.file_path)
    else:
        all_smiles = read_smiles(args.file_path)
        if not args.no_tqdm:
            print("Number of SMILES entered: ", len(all_smiles))
        
        cou = 0
        if args.save_mol:
            writer = Chem.SDWriter(args.save_dir + "/molecules.sdf")
        if args.no_tqdm:
            iterator = range(len(all_smiles))
        else:
            iterator = tqdm(range(len(all_smiles)), total=len(all_smiles), desc = 'Process 1/3')

        for i in iterator:
            try:
                mol = Chem.MolFromSmiles(all_smiles[i])
                mol = Chem.RemoveHs(mol)
                mols.append(mol)

                if args.save_mol:
                    writer.write(mol)
            except Exception as e:
                print(f"Error: {e}")
                cou += 1
            if not args.no_tqdm and i % 1000 == 0:
                iterator.set_postfix({'Error': cou})
        if args.save_mol:
            writer.close()
        if cou > 0:
            pass
            # raise ValueError("There might be some errors. Check your SMILES data.")

    # print("Process 2/9 is running", end = '...')
    fragmentizer = Fragmentizer(max_attach_atom_cnt=1)
    attachment_counter = {}
    motif_counter = Counter()
    # fragments = []
    atom_tokens = set()

    if args.no_tqdm:
        if suppl is not None:
            iterator = enumerate(suppl)
        else:
            iterator = enumerate(mols)
    else:
        if suppl is not None:
            iterator = tqdm(enumerate(suppl), total = len(suppl), mininterval=0.5, desc='Process 2/3')
        else:
            iterator = tqdm(enumerate(mols), total = len(mols), mininterval=0.5, desc='Process 2/3')

    success_cnt = 0
    total_cnt = 0
    error_smiles = []
    valid_smiles = []
    for i, m in iterator:
        try:
            smiles = Chem.MolToSmiles(copy.deepcopy(m), canonical=True)
            motifs, _ = fragmentizer.split_to_motif(m)
            for motif in motifs:
                if motif.smiles not in attachment_counter:
                    attachment_counter[motif.smiles] = {}
                att_tuple = motif.attachment.to_tuple()
                if att_tuple not in attachment_counter[motif.smiles]:
                    attachment_counter[motif.smiles][att_tuple] = 0
                attachment_counter[motif.smiles][att_tuple] += 1
                motif_counter[motif.smiles] += 1
            success_cnt += 1
            valid_smiles.append(smiles)
        except Exception as e:
            try:
                error_smiles.append(Chem.MolToSmiles(m))
            except:
                error_smiles.append(f'Error: {i}')
        finally:
            total_cnt += 1
            if not args.no_tqdm and total_cnt % 100 == 0:
                iterator.set_postfix_str(f'Success: {success_cnt}/{total_cnt} ({success_cnt/total_cnt:.2%})')

    preprecess_save_dir = os.path.join(args.save_dir, "preprocess")
    os.makedirs(preprecess_save_dir, exist_ok = True)
    if 'b' in  args.save_cnt_label:
        dill.dump(attachment_counter, open(preprecess_save_dir + "/attachment_counter.pkl", "wb"))
        dill.dump(valid_smiles, open(preprecess_save_dir + "/valid_smiles.pkl", "wb"))
        
    if 't' in  args.save_cnt_label:
        with open(preprecess_save_dir + "/attachment_counter.tsv", "w") as f:
            f.write("\n".join([str(smi)+'\t'+ str(motif_counter[smi]) + '\t' + '\t'.join([str(p)+'\t'+str(c) for p,c in v.items()]) for smi, v in attachment_counter.items()]))
        
        with open(preprecess_save_dir + "/valid_smiles.txt", "w") as f:
            f.write("\n".join(valid_smiles))

    if args.error_file is not None:
        with open(os.path.join(preprecess_save_dir, args.error_file), "w") as f:
            f.write("\n".join(error_smiles))

    distribution_df = plot_counter_distribution(motif_counter, save_file=os.path.join(args.save_dir, "preprocess", "plot", 'smiles_count_labels_0.png'), bin_width=1, y_scale='log', verbose=not args.no_tqdm)
    distribution_df.to_csv(os.path.join(preprecess_save_dir, "plot", 'motif_count_stats.tsv'), index=True, sep='\t')
    distribution_df = plot_counter_distribution(motif_counter, save_file=os.path.join(args.save_dir, "preprocess", "plot", 'vocab_count_labels_5.png'), bin_width=1, display_thresh=5, y_scale='log', verbose=not args.no_tqdm)
    
    if not args.no_tqdm:
        print('done')


# python vocab.py -f /workspaces/hgraph/mnt/Ms2z/data/SMILES/pubchem/pubchem_smiles_1M.pkl
# python vocab.py -f /workspaces/hgraph/mnt/Ms2z/data/SMILES/pubchem/pubchem_smiles_10k.pkl
if __name__ == "__main__":
    import warnings
    import argparse
    warnings.simplefilter('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--file_path", type = str, required=True, help = "Path of SMILES data for input compounds (delete SMILES containing '.')")
    parser.add_argument('-o', '--save_dir', type = str, required=True, help = "Path to save created data")
    parser.add_argument('--save_cnt_label', type = str, choices=['b', 't', 'bt', 'tb'], default = 't', help = "Type of count label to save")
    parser.add_argument('--save_mol', action='store_true', help = "Save the created molecule data with RDKit")
    parser.add_argument('--error_file', type = str, default = None, help = "Path to save SMILES data that caused errors")
    parser.add_argument('--no_tqdm', action='store_true', help = "Disable tqdm")

    args = parser.parse_args()
    main(args)