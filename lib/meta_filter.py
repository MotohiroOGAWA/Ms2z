from rdkit import Chem
from rdkit.Chem import Descriptors
import yaml
import os

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

from .msp_io import save_msp_data
from .chem_data import read_aduct_type_data


aduct_type_data = read_aduct_type_data()
def filter_aduct_type(metadata_df, aduct_type:list[str], column = "PrecursorType"):
    mask = metadata_df[column].isin(aduct_type)
    filtered_metadata = metadata_df[mask]
    return filtered_metadata


def filter_smiles(metadata_df, column="SMILES"):
    # mask = metadata_df[column].swifter.apply(lambda x: Chem.MolFromSmiles(x) is not None)
    mask = metadata_df[column].apply(lambda x: Chem.MolFromSmiles(x) is not None)
    filtered_metadata = metadata_df[mask]
    return filtered_metadata

counter_ions = [
    "O=S(=O)(O)O",
    "Cl",
    "[Cl-]",
    "Br",
    "[Br-]",
    "[Na+]",
]

counter_smiles_set = set(Chem.MolToSmiles(Chem.MolFromSmiles(counter_ion)) for counter_ion in counter_ions)

def remove_counter_ions(metadata_df, smiles_column="SMILES"):
    detected_counter_smiles = set()  
    noignore_smiles = []

    # Vectorized operation to handle SMILES strings containing "."
    def process_smiles(smiles):
        if "." in smiles:
            valid_smiles = []
            # Split the SMILES and check each fragment
            for s in smiles.split("."):
                mol = Chem.MolFromSmiles(s)
                if mol:
                    _s = Chem.MolToSmiles(mol)
                    if _s not in counter_smiles_set:
                        valid_smiles.append(s)
                    else:
                        detected_counter_smiles.add(_s)
            if len(valid_smiles) == 1:
                return valid_smiles[0]
            else:
                noignore_smiles.append(".".join(valid_smiles))
        return smiles

    # Apply the process_smiles function to the dataframe
    metadata_df[smiles_column] = metadata_df[smiles_column].apply(process_smiles)

    return metadata_df, list(detected_counter_smiles), noignore_smiles

def filter_have_counter_smiles(metadata_df, smiles_column="SMILES"):
    mask = metadata_df[smiles_column].str.contains("\\.")
    filtered_metadata = metadata_df[~mask]
    return filtered_metadata

def filter_precursor_mz(metadata_df, ppm=10, smiles_column="SMILES", aduct_type_column="PrecursorType", precursor_mz_column="PrecursorMZ"):
    tolerance = ppm / 1e6
    calculated_neutral_mass = []
    formula_list = []
    
    def process_precursor_mz(row):
        mol = Chem.MolFromSmiles(row[smiles_column])
        if mol:
            exact_mass = Descriptors.ExactMolWt(mol)
            precursor_mz = row[precursor_mz_column]
            aduct_type = row[aduct_type_column]
            aduct_mass = aduct_type_data[aduct_type]['shift']
            expected_mass = exact_mass + aduct_mass

            if abs(expected_mass - precursor_mz) < expected_mass*tolerance:
                formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
                formula_list.append(formula)
                calculated_neutral_mass.append(precursor_mz - aduct_mass)
                return True
        return False
    
    mask = metadata_df.apply(process_precursor_mz, axis=1)
    filtered_metadata = metadata_df[mask]
    filtered_metadata['NeutralMass'] = calculated_neutral_mass
    filtered_metadata['Formula'] = formula_list
    return filtered_metadata

def filter_workflow(peaks, metada_df, config_file, save_dir=None):
    config = yaml.safe_load(open(config_file, "r"))

    ppm = config['ppm'] # ppm 
    smiles_column = config.get('smiles_column', 'SMILES')
    aduct_type_column = config.get('aduct_type_column', 'PrecursorType')
    precursor_mz_column = config.get('precursor_mz_column', 'PrecursorMZ')
    aduct_types = config['aduct_types']

    print(f"Before filtering: {metada_df.shape}")
    filtered_metadata_df = metada_df

    print(f"Filtering aduct type: {filtered_metadata_df.shape} -> ", end="")
    filtered_metadata_df = filter_aduct_type(metada_df, aduct_types, aduct_type_column)
    print(f"{filtered_metadata_df.shape}")
    
    # print(f"Filtering SMILES: {filtered_metadata_df.shape} -> ", end="")
    # filtered_metadata_df = filter_smiles(filtered_metadata_df, column=smiles_column)
    # print(f"{filtered_metadata_df.shape}")

    print(f"Removing counter ions: {filtered_metadata_df.shape} -> ", end="")
    filtered_metadata_df, detected_counter_smiles, noignore_smiles = remove_counter_ions(filtered_metadata_df, smiles_column=smiles_column)
    print(f"{filtered_metadata_df.shape}")

    print(f"Filtering have counter SMILES: {filtered_metadata_df.shape} -> ", end="")
    filtered_metadata_df = filter_have_counter_smiles(filtered_metadata_df, smiles_column=smiles_column)
    print(f"{filtered_metadata_df.shape}")

    print(f"Filtering precursor mz: {filtered_metadata_df.shape} -> ", end="")
    filtered_metadata_df = filter_precursor_mz(filtered_metadata_df, ppm, smiles_column=smiles_column, aduct_type_column=aduct_type_column, precursor_mz_column=precursor_mz_column)
    print(f"{filtered_metadata_df.shape}")
    
    original_indices = filtered_metadata_df.index
    # Add original index to metadata if it does not exist
    if 'original_indices' not in filtered_metadata_df.columns:
        filtered_metadata_df['original_index'] = original_indices
    filtered_metadata_df = filtered_metadata_df.reset_index(drop=True)
    filtered_peaks = [peaks[i] for i in original_indices]
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_msp_data(filtered_peaks, filtered_metadata_df, save_dir)
        yaml.dump(config, open(os.path.join(save_dir, "filter_config.yaml"), "w"))

    return filtered_peaks, filtered_metadata_df



# if __name__ == '__main__':
#     metadata = pd.read_csv("metadata.csv")
#     filtered_metadata = filter_aduct_type(metadata, ["[M+H]+", "[M]+"])
#     print(filtered_metadata)