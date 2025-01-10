from lib.utils import *
from lib.fragmentizer import Fragmentizer
from lib.fragment import Fragment
from stats.plot_utils import plot_counter_distribution

from collections import Counter, defaultdict
from tqdm import tqdm
import os

from rdkit import Chem

def main(args):
    os.makedirs(args.save_dir, exist_ok = True)

    suppl = None
    mols = []
    if args.file_path.endswith('.sdf'):
        print("Reading Chem.Mol from SDF file")
        suppl = Chem.SDMolSupplier(args.file_path)
    else:
        all_smiles = read_smiles(args.file_path)
        print("Number of SMILES entered: ", len(all_smiles))
        
        cou = 0
        if args.save_mol:
            writer = Chem.SDWriter(args.save_dir + "/molecules.sdf")
        with tqdm(range(len(all_smiles)), total=len(all_smiles), desc = 'Process 1/3') as pbar:
            for i in pbar:
                try:
                    mol = Chem.MolFromSmiles(all_smiles[i])
                    mol = Chem.RemoveHs(mol)
                    mols.append(mol)

                    if args.save_mol:
                        writer.write(mol)
                except Exception as e:
                    print(f"Error: {e}")
                    cou += 1
                if i % 1000 == 0:
                    pbar.set_postfix({'Error': cou})
            if args.save_mol:
                writer.close()
        if cou > 0:
            pass
            # raise ValueError("There might be some errors. Check your SMILES data.")

    # print("Process 2/9 is running", end = '...')
    fragmentizer = Fragmentizer()
    count_labels = Counter() #(substructureSMILES,(AtomIdx in substructure, join order)xN)->frequency of use of label
    atom_counter = Counter()
    bond_counter = Counter()
    # fragments = []
    atom_tokens = set()

    if suppl is not None:
        iterator = tqdm(enumerate(suppl), total = len(suppl), mininterval=0.5, desc='Process 2/3')
    else:
        iterator = tqdm(enumerate(mols), total = len(mols), mininterval=0.5, desc='Process 2/3')

    success_cnt = 0
    total_cnt = 0
    error_smiles = []
    for i, m in iterator:
        try:
            monoatomic_tokens = Fragment.get_monoatomic_tokens(m)
            fragment_group = fragmentizer.split_molecule(m)
            atom_count = count_atoms_in_molecule(m)
            bond_count = count_bonds_in_molecule(m)
            count_labels.update([tuple(fragment) for fragment in fragment_group])
            atom_tokens.update(monoatomic_tokens)
            atom_counter.update(atom_count)
            bond_counter.update(bond_count)
            success_cnt += 1
        except Exception as e:
            try:
                error_smiles.append(Chem.MolToSmiles(m))
            except:
                error_smiles.append(f'Error: {i}')
        finally:
            total_cnt += 1
            if total_cnt % 100 == 0:
                iterator.set_postfix_str(f'Success: {success_cnt}/{total_cnt} ({success_cnt/total_cnt:.2%})')
    count_labels = dict(count_labels.most_common())
    atom_tokens = atom_tokens_sort(list(atom_tokens))

    smiles_counter = {}
    joint_counter = {}
    
    for frag_tuple, cnt in tqdm(count_labels.items(), desc='Process 3/3'):
        smiles = frag_tuple[0]
        if smiles not in smiles_counter:
            smiles_counter[smiles] = 0
            joint_counter[smiles] = {}
        smiles_counter[smiles] += cnt
        joint_list = frag_to_joint_list(frag_tuple)
        for joint in joint_list:
            if joint not in joint_counter[smiles]:
                joint_counter[smiles][joint] = 0
            joint_counter[smiles][joint] += cnt
    smiles_counter = dict(sorted(smiles_counter.items(), key=lambda x: x[1], reverse=True))
    joint_counter = {k: dict(sorted(v.items(), key=lambda x: x[1], reverse=True)) for k, v in joint_counter.items()}

    os.makedirs(args.save_dir + "/preprocess", exist_ok = True)
    if 'b' in  args.save_cnt_label:
        dill.dump(count_labels, open(args.save_dir + "/preprocess/fragment_counter.pkl", "wb"))
        
    if 't' in  args.save_cnt_label:
        with open(args.save_dir + "/preprocess/fragment_counter.txt", "w") as f:
            f.write("\n".join([str(k) + '\t' + str(v) for k, v in count_labels.items()]))

    with open(args.save_dir + "/preprocess/monoatomic_tokens.txt", "w") as f:
        f.write("\n".join([str(k) for k in atom_tokens]))

    with open(args.save_dir + "/preprocess/atom_count.txt", "w") as f:
        for key, value in atom_counter.items():
            f.write(f"{key}\t{value}\n")
    
    with open(args.save_dir + "/preprocess/bond_count.txt", "w") as f:
        for key, value in bond_counter.items():
            f.write(f"{key}\t{value}\n")

    with open(args.save_dir + "/preprocess/smiles_counter.tsv", "w") as f:
        for key, value in smiles_counter.items():
            f.write(f"{key}\t{value}\n")

    with open(args.save_dir + "/preprocess/joint_counter.tsv", "w") as f:
        for smiles, joint_counter in joint_counter.items():
            f.write(f"{smiles}\t")
            for key, value in joint_counter.items():
                f.write(f"{key}\t{value}\t")
            f.write("\n")

    if args.error_smi_file is not None:
        with open(os.path.join(args.save_dir, args.error_smi_file), "w") as f:
            f.write("\n".join(error_smiles))

    distribution_df = plot_counter_distribution(count_labels, save_file=os.path.join(args.save_dir, "preprocess", "plot", 'vocab_count_labels_0.png'), bin_width=1, y_scale='log')
    distribution_df.to_csv(os.path.join(args.save_dir, "preprocess", "plot", 'vocab_count_labels.tsv'), index=True, sep='\t')
    distribution_df = plot_counter_distribution(count_labels, save_file=os.path.join(args.save_dir, "preprocess", "plot", 'vocab_count_labels_5.png'), bin_width=1, display_thresh=5, y_scale='log')
    print('done')

def atom_tokens_sort(strings):
    """
    Sort atom tokens based on specific rules:
    - Split strings into three parts using regular expressions.
    - First part: Sorted using a custom sort order.
    - Second part: Sorted based on atomic number (from element symbol).
    - Third part: Sorted using custom order or alphabetically if not in order.
    
    Parameters:
        strings (list): List of strings to sort.
    
    Returns:
        list: Sorted list of strings.
    """
    
    def sort_key(s):
        # Use regex to split the string into three parts
        s = s.replace('(', '').replace(')', '')
        s = [a.strip().replace("'", "") for a in s.split(',')]
        try:
            atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(s[0])
        except:
            try:
                atom = Chem.MolFromSmiles(s[0]).GetAtomWithIdx(0)
                atomic_num = atom.GetAtomicNum()+0.5
            except:
                atomic_num = float('inf')
        bonds = s[2::2]
        bonds = [bond_priority.get(b, float('inf')) for b in bonds]
        
        if len(bonds) < 4:
            bonds += [-1] * (4 - len(bonds))
        
        return tuple([atomic_num]+bonds)
    
    # Sort the strings using the custom key
    return sorted(strings, key=sort_key)


def frag_to_joint_list(frag_tuple):
    atom_idx_to_bond_counter = defaultdict(lambda: [0, 0, 0])
    for atom_idx, bond_token in zip(frag_tuple[1::2], frag_tuple[2::2]):
        atom_idx_to_bond_counter[atom_idx][token_to_num_bond(bond_token)-1] += 1
    joint_list = [(atom_idx, bond_counter[0], bond_counter[1], bond_counter[2]) for atom_idx, bond_counter in atom_idx_to_bond_counter.items()]
    return joint_list
    

# python vocab.py -f /workspaces/hgraph/mnt/Ms2z/data/SMILES/pubchem/pubchem_smiles_1M.pkl
# python vocab.py -f /workspaces/hgraph/mnt/Ms2z/data/SMILES/pubchem/pubchem_smiles_10k.pkl
if __name__ == "__main__":
    import warnings
    import argparse
    warnings.simplefilter('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--file_path", type = str, required=True, help = "Path of SMILES data for input compounds (delete SMILES containing '.')")
    parser.add_argument('-o', '--save_dir', type = str, required=True, help = "Path to save created data")
    parser.add_argument('--save_cnt_label', type = str, choices=['b', 't', 'bt', 'tb'], default = 'b', help = "Type of count label to save")
    parser.add_argument('--save_mol', action='store_true', help = "Save the created molecule data with RDKit")
    parser.add_argument('--error_smi_file', type = str, default = None, help = "Path to save SMILES data that caused errors")

    args = parser.parse_args()
    main(args)