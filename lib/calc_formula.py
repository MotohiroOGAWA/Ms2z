import os
from rdkit import Chem
from rdkit.Chem import Descriptors
import itertools
import re
import bisect
from collections import defaultdict
import math
import concurrent.futures
from tqdm import tqdm
from tempfile import mkdtemp
import shutil
import pandas as pd

try:
    from .chem_data import read_adduct_type_data, calc_exact_mass
except ImportError:
    from chem_data import read_adduct_type_data, calc_exact_mass

adduct_type_data = read_adduct_type_data



def generate_possible_fragments(element_counts, unsaturation:int, max_unsaturation:int=999):
    """
    Generates all possible fragments by removing elements in all combinations
    while keeping the element counts non-negative.

    Args:
        element_counts (dict): A dictionary with elements and their counts.

    Returns:
        list: A list of dictionaries representing all possible fragments.
    """
    elements = list(element_counts.keys())
    if 'H' in elements:
        elements.remove('H')
    possible_fragments = []

    # Generate all combinations of elements' counts being reduced
    fragment = element_counts.copy()
    h_res = calc_hydrogen_for_each_unsaturation(fragment, 0, max_unsaturation)
    fragment['H'] = 0
    for h, u in h_res:
        fragment['H'] = h
        possible_fragments.append({k: v for k, v in fragment.items() if v > 0})

    for num_removed in range(1, len(elements) + 1):
        for combination in itertools.combinations(elements, num_removed):
            ranges = [range(1, element_counts[combination[i]] + 1) for i in range(num_removed)]
            for reduction in itertools.product(*ranges):
                fragment = element_counts.copy()
                for i, elem in enumerate(combination):
                    fragment[elem] -= reduction[i]
                h_res = calc_hydrogen_for_each_unsaturation(fragment, 0, max_unsaturation)
                fragment['H'] = 0
                for h, u in h_res:
                    fragment['H'] = h
                    possible_fragments.append({k: v for k, v in fragment.items() if v > 0})

    return possible_fragments

# マッチした(mz,intensity, formula)を返す
def annotate_peak_with_formula(peaks, formula, mol, aduct_type, ppm):
    aduct_mass = adduct_type_data[aduct_type]['shift']

    # Sort peaks by mass in descending order
    peaks = sorted(peaks, key=lambda x: x[0], reverse=False)
    max_intensity = max([peak[1] for peak in peaks])

    # Convert the formula into a dictionary of element counts
    element_counts = formula_to_dict(formula)
    ppm_threshold = calc_exact_mass(element_counts) * ppm * 1e-6

    # Calculate the unsaturation number for the formula
    unsaturation = calc_unsaturations(element_counts)

    # Generate all possible fragments from the formula
    possible_fragments = generate_possible_fragments(element_counts, unsaturation=unsaturation)
    fragment_mass_list = [calc_exact_mass(f) for f in possible_fragments]
    fragments_detail, fragment_mass_list = zip(*sorted(zip(possible_fragments, fragment_mass_list), key=lambda x: x[1]))
    peaks_with_formula = [(calc_exact_mass(element_counts), 1.1, formula, unsaturation, 0.0)]
    
    for peak_mass, intensity in peaks:
        p_idx_l = bisect.bisect_right(fragment_mass_list, peak_mass - aduct_mass - ppm_threshold)
        p_idx_r = bisect.bisect_left(fragment_mass_list, peak_mass - aduct_mass + ppm_threshold)
                
        min_ppm = math.inf
        min_idx = -1
        for i in range(p_idx_l, p_idx_r):
            fragment = fragments_detail[i]
            ppm = abs(peak_mass - aduct_mass - fragment_mass_list[i]) / ppm_threshold
            if ppm < min_ppm:
                min_ppm = ppm
                min_idx = i
                match_formula = dict_to_formula(fragment)
        if min_idx != -1:
            unsaturation = calc_unsaturations(fragments_detail[min_idx])
            peaks_with_formula.append((peak_mass, intensity/max_intensity, match_formula, unsaturation, min_ppm))

    calc_formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    if formula != calc_formula:
        raise ValueError(f"Formula mismatch: {formula} != {calc_formula}")
    
    return ";".join([",".join([str(peak_mass),str(intensity),formula, str(unsaturation), str(ppm)]) for peak_mass, intensity, formula, unsaturation, ppm in peaks_with_formula])



# 詳しくフラグメントを取得
def annotate_peak_with_formula_detail(peaks, formula, mol, aduct_type, ppm):
    """
    Annotates peaks with the formula if the peak's mass is within 10 ppm of any possible fragment's exact mass.

    Args:
        peaks (list of tuples): List of peaks, where each peak is a tuple (mass, intensity).
        formula (str): The chemical formula to compare the peaks against.

    Returns:
        list: A list of tuples where each tuple contains the peak (mass, intensity) and the corresponding formula
              fragment if the peak's mass matches the fragment's exact mass within 10 ppm.
    """
    aduct_mass = adduct_type_data[aduct_type]['shift']

    # Sort peaks by mass in descending order
    peaks = sorted(peaks, key=lambda x: x[0], reverse=False)
    max_intensity = max([peak[1] for peak in peaks])

    # Convert the formula into a dictionary of element counts
    element_counts = formula_to_dict(formula)
    ppm_threshold = calc_exact_mass(element_counts) * ppm * 1e-6

    # Calculate the unsaturation number for the formula
    unsaturation = calc_unsaturations(element_counts)

    # Generate all possible fragments from the formula
    possible_fragments = generate_possible_fragments(element_counts, unsaturation=unsaturation)
    fragment_mass_list = [calc_exact_mass(f) for f in possible_fragments]
    fragments_detail, fragment_mass_list = zip(*sorted(zip(possible_fragments, fragment_mass_list), key=lambda x: x[1]))
    peaks_with_formula = []
    

    for peak_mass, intensity in peaks:
        p_idx_l = bisect.bisect_right(fragment_mass_list, peak_mass - aduct_mass - ppm_threshold)
        p_idx_r = bisect.bisect_left(fragment_mass_list, peak_mass - aduct_mass + ppm_threshold)
        detail_list = []
        
        for i in range(p_idx_l, p_idx_r):
            fragment = fragments_detail[i]
            ppm = abs(peak_mass - aduct_mass - fragment_mass_list[i]) / ppm_threshold
            detail_list.append((dict_to_formula(fragment), ppm))
        peaks_with_formula.append((peak_mass, intensity/max_intensity, detail_list))

    calc_formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    if formula != calc_formula:
        raise ValueError(f"Formula mismatch: {formula} != {calc_formula}")
    

    formula_to_peak_idx = {}
    for i, peak in enumerate(peaks_with_formula):
        if len(peak[2]) > 0:
            for j in range(len(peak[2])):
                formula_to_peak_idx[peak[2][j][0]] = [i,j]
    
    # candidate_fragments = get_all_fragments(mol, list(formula_to_peak_idx.keys()))

    fragments = []
    for p in peaks_with_formula:
        _smiles = []
        # for f in p[2]:
        #     _formula = f[0]
        #     if _formula in candidate_fragments:
        #         _smiles.append(candidate_fragments[_formula])
        #     else:
        #         _smiles.append(None)
        fragments.append((p[0],p[1],p[2],_smiles))
    
    return fragments


def process_batch_file(batch_file, ppm, smiles_column, peaks_column, formula_column, aduct_type_column, mol_list):
    try:
        # Load the batch file
        batch_df = pd.read_parquet(batch_file)
        
        # Annotate the peaks for each row in the batch
        annotated_peaks = []
        for idx, row in batch_df.iterrows():
            peaks = row[peaks_column]
            formula = row[formula_column]
            aduct_type = row[aduct_type_column]
            
            # Parse peaks (assumes they are formatted as 'mass,intensity;mass,intensity;...')
            peaks = [(float(peak_mass), float(intensity)) for peak_mass, intensity in [peak.split(',') for peak in peaks.split(';')]]
            
            # Annotate the peaks
            annotated_peak = annotate_peak_with_formula(peaks=peaks, formula=formula, mol=mol_list[idx], aduct_type=aduct_type, ppm=ppm)
            annotated_peaks.append(annotated_peak)
        
        # Add the annotated peaks to the DataFrame
        batch_df['FormulaPeaks'] = annotated_peaks

        # Save the annotated batch back to a file
        batch_df.to_parquet(batch_file, index=False)
        return batch_file

    except Exception as e:
        # print(f"Error processing batch file {batch_file}: {e}")
        return None

def annotate_peak_with_formula_df(spectra_df, ppm, mol_list=None, ncpu=None, batch_size=1000, smiles_column='SMILES', peaks_column='Peak', formula_column='Formula', aduct_type_column='PrecursorType'):
    """
    Annotates peaks in batches with the formula if the peak's mass is within 10 ppm of any possible fragment's exact mass.
    
    The data is split into batches and processed in parallel. Results are saved to a temporary directory and combined
    once all processing is finished. The temporary directory is deleted after use.

    Args:
        spectra_df (pandas.DataFrame): Dataframe containing the spectra information.
        ppm (int): The mass tolerance in ppm.
        mol_list (list): List of RDKit Mol objects corresponding to the SMILES strings in the dataframe.
        batch_size (int): Size of each batch.
        ncpu (int, optional): Number of CPU cores to use for parallel processing. If None, it will use all available cores.

    Returns:
        pandas.DataFrame: Dataframe with the annotated peaks.
    """
    if mol_list is None:
        mol_list = [Chem.MolFromSmiles(smiles) for smiles in spectra_df[smiles_column]]

    # Create a temporary directory for batch processing
    temp_dir = mkdtemp()

    try:
        # Split the DataFrame into batches and save them to the temporary directory
        batch_files = []
        for i in range(0, len(spectra_df), batch_size):
            batch_df = spectra_df.iloc[i:i+batch_size]
            batch_file = os.path.join(temp_dir, f'batch_{i}.parquet')
            batch_df.to_parquet(batch_file, index=False)
            batch_files.append(batch_file)

        # Process each batch in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=ncpu) as executor:
            futures = {executor.submit(process_batch_file, batch_file, ppm, smiles_column, peaks_column, formula_column, aduct_type_column, mol_list): batch_file for batch_file in batch_files}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                future.result()  # Wait for all batch processing to complete

        # Combine all the processed batch files into one DataFrame
        processed_dfs = [pd.read_parquet(batch_file) for batch_file in batch_files]

        # Concatenate all DataFrames
        final_df = pd.concat(processed_dfs, ignore_index=True)

        return final_df

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)

def get_all_fragments(mol: Chem.Mol, formulas:list[str]):
    if mol is None:
        raise ValueError("Invalid Mol object")
    
    numbond_to_counts = defaultdict(list)
    for formula in formulas:
        element_counts = formula_to_dict(formula)
        tmp = element_counts.copy()
        if 'H' in tmp:
            tmp.pop('H')
        numbond_to_counts[sum(tmp.values())].append(element_counts)

    paths = []
    start_atom_idx_list = []
    ring_info = mol.GetRingInfo()
    # Get the atom rings and bond rings
    atom_rings = ring_info.AtomRings()
    bond_rings = ring_info.BondRings()
    def dfs(current_atom_idx, current_path, remaining_depth):
        # Base case: If no more depth left, store the path and return
        if remaining_depth == 0:
            path_set = set(current_path)
            for p in current_path:
                atom = mol.GetAtomWithIdx(p)
                if atom.GetIsAromatic():
                    for ring in atom_rings:
                        if p in ring:
                            if any(mol.GetAtomWithIdx(r).GetIsAromatic() for r in ring):
                                if not path_set.issuperset(ring):
                                    return

            paths.append(current_path)
            return

        # Get the current atom
        current_atom = mol.GetAtomWithIdx(current_atom_idx)

        # Explore neighbors (atoms that are one bond away)
        for neighbor in current_atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()

            # Avoid cycles: Do not revisit atoms that are already in the current path
            if neighbor_idx not in current_path and neighbor_idx not in start_atom_idx_list:
                # Continue DFS with the neighbor
                dfs(neighbor_idx, current_path + [neighbor_idx], remaining_depth - 1)

    
    fragments = defaultdict(list)
    for atom_num, element_counts_list in numbond_to_counts.items():
        paths = []
        start_atom_idx_list = []
        for start_atom_idx in range(mol.GetNumAtoms()):
            start_atom_idx_list.append(start_atom_idx)
            dfs(start_atom_idx, [start_atom_idx], atom_num-1)

        for path in paths:
            bond_indices = []
            for i in range(len(path) - 1):
                bond = mol.GetBondBetweenAtoms(path[i], path[i + 1])
                if bond:
                    bond_indices.append(bond.GetIdx())

            # Check if atoms in the path form part of a ring
            for atom_ring, bond_ring in zip(atom_rings, bond_rings):  # Iterate over atom rings and corresponding bond rings
                if set(path).intersection(atom_ring):  # Check if any atom from path is in the ring
                    # If path contains atoms in the ring, include all bonds in that ring
                    bond_indices.extend(bond_ring)
            bond_indices = list(set(bond_indices))

            fragment = Chem.PathToSubmol(mol, bond_indices)
            fragment = Chem.RemoveHs(Chem.AddHs(fragment))
            formula = Chem.rdMolDescriptors.CalcMolFormula(fragment)
            smiles = Chem.MolToSmiles(fragment)
            if smiles not in fragments[formula]:
                fragments[formula].append(smiles)
    return fragments

def smiles_with_atom_indices(mol: Chem.Mol) -> str:
    """
    Generates a SMILES string with atom indices embedded as atom map numbers.
    
    :param mol: RDKit Mol object representing the molecule.
    :return: SMILES string with atom indices.
    """
    # Make a copy of the molecule to avoid modifying the original molecule
    mol_with_idx = Chem.Mol(mol)
    
    # Set atom map numbers to the atom index
    for atom in mol_with_idx.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())  # Set atom map number to the index

    # Generate a SMILES string with atom map numbers
    smiles_with_idx = Chem.MolToSmiles(mol_with_idx)
    
    return smiles_with_idx

def get_formula(mol: Chem.Mol) -> str:
    """
    Generate the molecular formula for a given RDKit molecule.
    
    :param mol: RDKit Mol object representing the molecule.
    :return: Molecular formula string.
    """
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    return formula

def formula_to_dict(formula):
    """
    Converts a chemical formula in string format into a dictionary with element symbols as keys
    and their respective counts as values.

    Args:
        formula (str): The chemical formula (e.g., 'C6H12O6').

    Returns:
        dict: A dictionary with elements as keys and their counts as values (e.g., {'C': 6, 'H': 12, 'O': 6}).
    """
    # Use regular expressions to find all element and count pairs in the formula
    # The pattern looks for an uppercase letter followed by an optional lowercase letter for the element
    # and then an optional number for the count
    matches = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    
    # Initialize an empty dictionary to store element counts
    element_counts = {}
    
    # Iterate over each element and its count
    for element, count in matches:
        # If no number is provided after the element, set the count to 1
        if count == '':
            count = 1
        else:
            # Convert the count to an integer if it is provided
            count = int(count)
        
        # Add the element and its count to the dictionary
        # If the element already exists, increment its count by the new value
        element_counts[element] = element_counts.get(element, 0) + count
    
    # Return the dictionary with the element counts
    return element_counts

def dict_to_formula(element_counts):
    """
    Converts a dictionary with element symbols as keys and their respective counts as values
    into a chemical formula string, ensuring that the formula is ordered according to the Hill system.

    Args:
        element_counts (dict): A dictionary with elements as keys and their counts as values.
                                Example: {'C': 6, 'H': 12, 'O': 6}

    Returns:
        str: The chemical formula string (e.g., 'C6H12O6').
    """
    # Initialize an empty list to store the ordered element and count pairs
    formula_list = []
    
    # First, add 'C' (if present) followed by 'H' (if present)
    element_counts = element_counts.copy()
    if 'C' in element_counts:
        count = element_counts.pop('C')
        if count == 1:
            formula_list.append('C')
        else:
            formula_list.append(f'C{count}')
    
    if 'H' in element_counts:
        count = element_counts.pop('H')
        if count == 1:
            formula_list.append('H')
        else:
            formula_list.append(f'H{count}')
    
    # Sort the remaining elements alphabetically
    for element in sorted(element_counts):
        count = element_counts[element]
        if count == 1:
            formula_list.append(element)
        else:
            formula_list.append(f'{element}{count}')
    
    # Join the element and count pairs into a single string to form the chemical formula
    return ''.join(formula_list)

def calc_unsaturations(element_counts):
    """
    Calculate the unsaturation number (degree of unsaturation) based on the given element counts.

    Args:
        element_counts (dict): Dictionary containing the counts of each element. Example: {'C': 6, 'H': 12, 'O': 6}

    Returns:
        int: The unsaturation number.
    """
    # Calculate the unsaturation number using the formula: U = C - (H/2) + (N/2) + 1
    # where C, H, and N are the counts of carbon, hydrogen, and nitrogen, respectively
    C = element_counts.get('C', 0)
    H = element_counts.get('H', 0)
    N = element_counts.get('N', 0)
    P = element_counts.get('P', 0)
    X = sum([element_counts.get(x, 0) for x in ['F', 'Cl', 'Br', 'I']])  # Halogens
    unsaturation = (2 * C + 2 - X + N + P - H) / 2

    if int(unsaturation/0.5) % 2 == 0:
        return int(unsaturation)
    else:
        return unsaturation

def calc_hydrogen_from_unsaturation(element_counts, unsaturation):
    """
    Calculate the number of hydrogens required to achieve a specific unsaturation level
    based on the given elements and unsaturation value.

    Args:
        element_counts (dict): Dictionary containing the counts of each element. Example: {'C': 6, 'H': 12, 'O': 6}
        unsaturation (int): The desired unsaturation value.

    Returns:
        int: The number of hydrogens required to achieve the specified unsaturation level.
    """
    # Calculate the required hydrogen count for the given unsaturation level
    C = element_counts.get('C', 0)
    N = element_counts.get('N', 0)
    P = element_counts.get('P', 0)
    X = sum([element_counts.get(x, 0) for x in ['F', 'Cl', 'Br', 'I']])  # Halogens
    h = 2 * C + 2 - X + N + P - int(2 * unsaturation)
    
    return h

def calc_hydrogen_for_each_unsaturation(element_counts, min_unsaturation, max_unsaturation, int_unsaturation=True):
    """
    Calculate the number of hydrogens required to achieve an integer unsaturation level
    based on the given elements and unsaturation range.

    Args:
        element_counts (dict): Dictionary containing the counts of each element. Example: {'C': 6, 'H': 12, 'O': 6}
        min_unsaturation (int): Minimum unsaturation value.
        max_unsaturation (int): Maximum unsaturation value.
        int_unsaturation (bool): Whether to calculate hydrogen counts for integer unsaturation only. Default is True.

    Returns:
        list: A list of tuples, where each tuple contains the hydrogen count and the corresponding unsaturation.
    """
    
    results = []
    
    # Get counts for relevant elements, default to 0 if element not present
    # C = element_counts.get('C', 0)
    H = element_counts.get('H', 0)
    # N = element_counts.get('N', 0)
    # X = sum([element_counts.get(x, 0) for x in ['F', 'Cl', 'Br', 'I']])  # Halogens
    
    # Iterate over possible unsaturation values
    for unsaturation in range(min_unsaturation, max_unsaturation + 1):
        if int_unsaturation:
            u_list = [unsaturation]
        elif unsaturation+0.5<=max_unsaturation:
            u_list = [unsaturation, unsaturation+0.5]
        else:
            u_list = [unsaturation]
        for unsaturation in u_list:
            # Calculate the required hydrogen count for the current unsaturation level
            # h = 2 * C + 2 - X + N - int(2 * unsaturation)
            h = calc_hydrogen_from_unsaturation(element_counts, unsaturation)
            
            # Only consider valid hydrogen counts (non-negative integers)
            if h < 0:
                continue
            if h > H:  # Ensure H is an even number
                continue
            results.append((h, unsaturation))
        
        if h < 0:
            break
    
    return results


# Function to fragment the molecule and retrieve SMILES and exact masses for each fragment
def get_fragments_and_masses(mol, peaks_with_formula, precursor_mz, aduct_type, ppm=10, threshold=0.2):
    tolerance = precursor_mz * ppm / 1e6
    aduct_mass = adduct_type_data[aduct_type]['shift']
    max_intensity = max([peak[1] for peak in peaks])

    candi_idx = [i for i, peak in enumerate(peaks) if peak[1]/max_intensity > threshold]
    high_idx = []

    bonds = mol.GetBonds()
    fragments = []
    
    for bond in bonds:
        # Skip the bond if it is part of a ring
        if bond.IsInRing():
            continue
        
        # Break the bond
        bond_index = bond.GetIdx()
        fragmented_mol = Chem.FragmentOnBonds(mol, [bond_index])
        
        # Get fragments in SMILES format
        frags = Chem.GetMolFrags(fragmented_mol, asMols=True)
        
        for frag in frags:
            frag = Chem.DeleteSubstructs(frag, Chem.MolFromSmiles('[*]'))
            # frag = Chem.AddHs(frag, addCoords=True)
            smiles_frag = Chem.MolToSmiles(frag)
            exact_mass = Descriptors.ExactMolWt(frag)

            for i, peak in enumerate(peaks):
                shift = peak[0] - aduct_mass
                if abs(shift - exact_mass) < tolerance:
                    fragments.append((peak[0], peak[1] / max_intensity,  smiles_frag, exact_mass, "PI"))
                    # print(f'SMILES(PI): {smiles_frag}, Exact Mass: {exact_mass:.4f}, Peak: {peak[0]}, Intensity: {peak[1]}, ppm: {abs(shift - exact_mass)/tolerance}')
                    if i in candi_idx and i not in high_idx:
                        high_idx.append(i)
                shift = precursor_mz - peak[0] 
                if abs(shift - exact_mass) < tolerance:
                    fragments.append((peak[0], peak[1] / max_intensity,  smiles_frag, exact_mass, "PI"))
                    # print(f'SMILES(NL): {smiles_frag}, Exact Mass: {exact_mass:.4f}, Peak: {peak[0]}, Intensity: {peak[1]}, ppm: {abs(shift - exact_mass)/tolerance}')
                    if i in candi_idx and i not in high_idx:
                        high_idx.append(i)

            # fragments.append((peak[0], intensity, smiles_frag, exact_mass, bond_index, ))

    # # フラグメントとその正確質量を取得
    # fragments = get_fragments_and_masses(mol, peaks, precursor_mz, precursor_type)

    if len(fragments) == 0:
        return fragments, 0
    else:
        return fragments, len(high_idx) / len(candi_idx)


if __name__ == '__main__':
    if False:
        # 入力SMILESを指定
        smiles = 'OCC(=CCNC=1N=CNC2=NC=NC21)C'
        peaks = [[ 41.0399  ,   6.207207],
            [ 43.0192  ,  49.711712],
            [ 43.0766  ,   1.986987],
            [ 55.0301  ,   2.316316],
            [ 57.0709  ,   1.600601],
            [ 65.015   ,   3.496496],
            [ 65.0366  ,   1.581582],
            [ 67.0302  ,   6.727728],
            [ 67.0517  ,   4.393393],
            [ 82.0409  ,   2.598599],
            [ 92.0251  ,  16.601602],
            [ 94.0407  ,   8.375375],
            [109.0512  ,   5.252252],
            [119.0354  , 100.      ],
            [119.1288  ,   4.228228],
            [121.0556  ,   1.025025],
            [136.0619  ,  41.038038],
            [136.1628  ,   1.571572],
            [148.0621  ,   2.447447]]
        precursor_type = "[M+H]+"
        aduct_mass = 1.00782503223
        precursor_mz = 220.119293
        '''
        Name                                     Zeatin
        Synon                             $:00in-source
        DB#                           VF-NPL-QTOF009997
        InChIKey            UZKQTCBAMSWPJD-FARCUNLSSA-N
        PrecursorType                            [M+H]+
        SpectrumType                                MS2
        PrecursorMZ                          220.119293
        InstrumentType                      LC-ESI-QTOF
        Instrument                   Agilent 6530 Q-TOF
        IonMode                                       P
        CollisionEnergy1                             40
        CollisionEnergy2                             40
        Formula                               C10H13N5O
        MW                                          219
        ExactMass                            219.112015
        Comments                                       
        SMILES              OCC(=CCNC=1N=CNC2=NC=NC21)C
        NumPeaks                                     19
        NeutralMass                          219.111468
        original_index                              205
        Name: 0, dtype: object
        '''
        mol = Chem.MolFromSmiles(smiles)

        # エラーを防ぐためにKekulizeを適用
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Chem.KekulizeException as e:
            print(f"Kekulize failed: {e}")

        # フラグメントとその正確質量を取得
        fragments = get_fragments_and_masses(mol, peaks, precursor_mz, precursor_type)

        # 質量順にソート
        fragments_sorted = sorted(fragments, key=lambda x: x[1])

        # 結果を表示
        print('\nFragments')
        for frag_smiles, frag_mass in fragments_sorted:
            print(f'SMILES: {frag_smiles:30}, Exact Mass: {frag_mass:.4f}')

        print('\nNeutral loss')
        for frag_smiles, frag_mass in fragments_sorted:
            print(f'SMILES: {frag_smiles:30}, Exact Mass: {precursor_mz-frag_mass+aduct_mass:.4f}')
    elif True:
        formula = 'C10H13N5O'
        smiles = 'OCC(=CCNC=1N=CNC2=NC=NC21)C'
        peaks = [[ 41.0399  ,   6.207207],
            [ 43.0192  ,  49.711712],
            [ 43.0766  ,   1.986987],
            [ 55.0301  ,   2.316316],
            [ 57.0709  ,   1.600601],
            [ 65.015   ,   3.496496],
            [ 65.0366  ,   1.581582],
            [ 67.0302  ,   6.727728],
            [ 67.0517  ,   4.393393],
            [ 82.0409  ,   2.598599],
            [ 92.0251  ,  16.601602],
            [ 94.0407  ,   8.375375],
            [109.0512  ,   5.252252],
            [119.0354  , 100.      ],
            [119.1288  ,   4.228228],
            [121.0556  ,   1.025025],
            [136.0619  ,  41.038038],
            [136.1628  ,   1.571572],
            [148.0621  ,   2.447447]]
        annotate_peak_with_formula(peaks, formula, smiles, ppm=10, aduct_type="[M+H]+")
