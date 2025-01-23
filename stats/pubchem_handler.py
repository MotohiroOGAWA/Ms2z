import os
import pandas as pd
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


Property = [
    'MolecularFormula',
    'MolecularWeight',
    'InChI',
    'InChIKey',
    'IUPACName',
    'Title',
    'XLogP',
    'ExactMass',
    'MonoisotopicMass',
    'TPSA',
    'Complexity',
    'Charge',
    'HBondDonorCount',
    'HBondAcceptorCount',
    'RotatableBondCount',
    'HeavyAtomCount',
    'IsotopeAtomCount',
    'AtomStereoCount',
    'DefinedAtomStereoCount',
    'UndefinedAtomStereoCount',
    'BondStereoCount',
    'DefinedBondStereoCount',
    'UndefinedBondStereoCount',
    'CovalentUnitCount',
    'PatentCount',
    'PatentFamilyCount',
    'LiteratureCount',
    'Volume3D',
    'XStericQuadrupole3D',
    'YStericQuadrupole3D',
    'ZStericQuadrupole3D',
    'FeatureCount3D',
    'FeatureAcceptorCount3D',
    'FeatureDonorCount3D',
    'FeatureAnionCount3D',
    'FeatureCationCount3D',
    'FeatureRingCount3D',
    'FeatureHydrophobeCount3D',
    'ConformerModelRMSD3D',
    'EffectiveRotorCount3D',
    'ConformerCount3D',
    'Fingerprint2D'
]

def read_file(file_path, column_names, column_types=None, chunk_size=1000, show_progress=True):
    """
    Reads a tab-separated file and converts it into a pandas DataFrame.
    Optionally displays a progress bar while processing.

    Parameters:
        file_path (str): Path to the input file.
        column_names (list): List of column names for the DataFrame.
        column_types (dict): Optional dictionary specifying the data type for each column.
        chunk_size (int): Number of rows to process at a time (default: 1000).
        show_progress (bool): Whether to display the progress bar (default: True).

    Returns:
        pd.DataFrame: A DataFrame containing the data from the file.
    """
    try:
        # If progress is enabled, calculate total lines
        total_lines = sum(1 for _ in open(file_path, 'r')) if show_progress else None

        # Initialize an empty list to store chunks
        chunks = []

        # Determine the progress handler
        if show_progress:
            progress_handler = tqdm(total=total_lines, desc="Processing rows", unit="rows")
        else:
            progress_handler = None

        # Read the file in chunks
        for chunk in pd.read_csv(
            file_path,
            sep='\t',
            header=None,
            names=column_names,
            chunksize=chunk_size
        ):
            # Ensure data types are consistent
            if column_types:
                for col, dtype in column_types.items():
                    if col in chunk.columns:
                        chunk[col] = chunk[col].astype(dtype)
                    else:
                        chunk[col] = chunk[col].astype(str)
            if 'CID' in chunk.columns:
                chunk['CID'] = chunk['CID'].astype(str)

            # Append the chunk to the list
            chunks.append(chunk)

            # Update the progress bar if enabled
            if show_progress:
                progress_handler.update(len(chunk))

        # Close the progress bar
        if show_progress:
            progress_handler.close()

        # Combine all chunks into a single DataFrame and return
        return pd.concat(chunks, ignore_index=True)

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def CidSmilesToDataFrame(file_path, show_progress=True):
    """
    Reads a CID-SMILES file and converts it into a pandas DataFrame.

    Parameters:
        file_path (str): Path to the CID-SMILES file.
        show_progress (bool): Whether to display the progress bar.

    Returns:
        pd.DataFrame: A DataFrame containing CID and SMILES columns.
    """
    return read_file(file_path, column_names=["CID", "SMILES"], show_progress=show_progress)

def CidMassToDataFrame(file_path, show_progress=True):
    """
    Reads a CID-Mass file and converts it into a pandas DataFrame.

    Parameters:
        file_path (str): Path to the CID-Mass file.
        show_progress (bool): Whether to display the progress bar.

    Returns:
        pd.DataFrame: A DataFrame containing CID, Formula, MonoisotopicMass, and ExactMass columns.
    """
    column_types = {
        "MonoisotopicMass": "float32",
        "ExactMass": "float32"
    }
    return read_file(
        file_path,
        column_names=["CID", "Formula", "MonoisotopicMass", "ExactMass"],
        column_types=column_types,
        show_progress=show_progress
    )

def merge_files_by_cid(file_path_list, column_name_mapping, column_type_mapping=None, chunk_size=1000, show_progress=True):
    """
    Merges multiple files by the 'CID' column into a single DataFrame.

    Parameters:
        file_path_list (list): List of file paths to read and merge.
        column_name_mapping (dict): Dictionary mapping file paths to their respective column names.
        column_type_mapping (dict): Optional dictionary mapping file paths to column types (default: None).
        chunk_size (int): Number of rows to process at a time (default: 1000).
        show_progress (bool): Whether to display the progress bar (default: True).

    Returns:
        pd.DataFrame: A merged DataFrame combining all files on the 'CID' column.
    """
    merged_df = None

    for file_path in tqdm(file_path_list, desc="Processing files", unit="file"):
        # Get column names and types for the current file
        column_names = column_name_mapping[file_path]
        column_types = column_type_mapping[file_path] if column_type_mapping and file_path in column_type_mapping else None

        # Read the file
        if file_path.endswith('.parquet'):
            file_df = pd.read_parquet(file_path)
        else:
            file_df = read_file(file_path, column_names=column_names, column_types=column_types, chunk_size=chunk_size, show_progress=show_progress)

        if file_df is not None:
            if merged_df is None:
                merged_df = file_df
            else:
                # Merge on the 'CID' column
                merged_df = pd.merge(merged_df, file_df, on="CID", how="outer")

    return merged_df


def fetch_complexity(base_url, cid_list, max_retries):
    """
    Fetch Complexity for a list of CIDs from PubChem.

    Args:
        base_url (str): The base URL for the API.
        cid_list (list): List of CIDs to fetch.
        max_retries (int): Maximum number of retries for failed requests.

    Returns:
        dict: Dictionary where keys are CIDs and values are Complexity values.
    """
    results = {}
    for cid in cid_list:
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{base_url}?cid={cid}")
                response.raise_for_status()
                data = response.json()
                results[str(cid)] = data["PropertyTable"]["Properties"][0]["Complexity"]
                break
            except requests.exceptions.RequestException:
                if attempt == max_retries - 1:
                    results[str(cid)] = None
            except (KeyError, IndexError):
                results[str(cid)] = None
    return results

def get_complexity_from_cid(cid_list, max_retries=3, chunk_size=10, num_workers=5):
    """
    Retrieve Complexity values for a list of CIDs from PubChem using parallel requests.

    Args:
        cid_list (list): List of CIDs (integers or strings).
        max_retries (int): Maximum number of retries for failed requests.
        chunk_size (int): Number of CIDs to process per chunk.
        num_workers (int): Number of parallel workers.

    Returns:
        dict: Dictionary where keys are CIDs and values are Complexity values.
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/Complexity/JSON"

    results = {}
    failed_cids = []

    # Process in chunks
    for i in range(0, len(cid_list), chunk_size):
        chunk = cid_list[i:i + chunk_size]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_chunk = {executor.submit(fetch_complexity, base_url, chunk, max_retries): chunk for chunk in [chunk]}

            for future in as_completed(future_to_chunk):
                try:
                    chunk_results = future.result()
                    results.update(chunk_results)
                    failed_cids.extend([cid for cid, complexity in chunk_results.items() if complexity is None])
                except Exception as e:
                    failed_cids.extend(chunk)

    # Retry failed CIDs
    if failed_cids:
        print(f"Retrying for failed CIDs: {failed_cids}")
        retry_results = get_complexity_from_cid(failed_cids, max_retries, chunk_size, num_workers)
        results.update(retry_results)

    return results

def fetch_properties(base_url, cid_list, max_retries):
    """
    Fetch properties for a list of CIDs from PubChem.

    Args:
        base_url (str): The base URL for the API.
        cid_list (list): List of CIDs to fetch.
        max_retries (int): Maximum number of retries for failed requests.

    Returns:
        dict: Dictionary where keys are CIDs and values are dictionaries of property values.
    """
    results = {}
    for attempt in range(max_retries):
        try:
            cid_list_str = ",".join(map(str, cid_list))
            response = requests.get(base_url.format(cid_list_str))
            response.raise_for_status()
            data = response.json()
            for property in data["PropertyTable"]["Properties"]:
                cid = property["CID"]
                results[str(cid)] = {}
                for key, value in property.items():
                    if key != "CID":
                        results[str(cid)][key] = value
            break
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                results[str(cid)] = None
        except (KeyError, IndexError):
            results[str(cid)] = None
    return results

def get_properties_from_cid(cid_list, max_retries=3, chunk_size=1000, num_workers=5):
    """
    Retrieve properties for a list of CIDs from PubChem using parallel requests.

    Args:
        cid_list (list): List of CIDs (integers or strings).
        max_retries (int): Maximum number of retries for failed requests.
        chunk_size (int): Number of CIDs to process per chunk.
        num_workers (int): Number of parallel workers.

    Returns:
        dict: Dictionary where keys are CIDs and values are dictionaries of property values.
    """
    # base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/MolecularWeight,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,ExactMass,MonoisotopicMass,TPSA,HeavyAtomCount,FormalCharge,Complexity,IsotopeAtomCount,DefinedAtomStereoCount,UndefinedAtomStereoCount,DefinedBondStereoCount,UndefinedBondStereoCount,CovalentUnitCount,IsCanonicalized/JSON"
    properties = ','.join(Property)
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/"+properties+"/JSON"

    results = {}
    failed_cids = []

    # Process in chunks
    for i in range(0, len(cid_list), chunk_size):
        chunk = cid_list[i:i + chunk_size]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_chunk = {executor.submit(fetch_properties, base_url, chunk, max_retries): chunk for chunk in [chunk]}

            for future in as_completed(future_to_chunk):
                try:
                    chunk_results = future.result()
                    results.update(chunk_results)
                    failed_cids.extend([cid for cid, props in chunk_results.items() if props is None])
                except Exception as e:
                    failed_cids.extend(chunk)

    # Retry failed CIDs
    if failed_cids:
        print(f"Retrying for failed CIDs: {failed_cids}")
        retry_results = get_properties_from_cid(failed_cids, max_retries, chunk_size, num_workers)
        results.update(retry_results)

    return results

if __name__ == '__main__':
    process_name = 'MassToDataFrame'
    process_name = 'Merge'
    process_name = 'Property'

    if process_name == 'SMILEStoDataFrame':
        print('Reading CID-SMILES file...', end=' ')
        file_path = '/workspaces/Ms2z/mnt/data/smiles/pubchem/CID-SMILES'
        smiles_df = CidSmilesToDataFrame(file_path)
        print(f'Number of rows: {len(smiles_df)}')
        print('Save the DataFrame as a Parquet file...', end=' ')
        smiles_df.to_parquet(os.path.join(os.path.dirname(file_path), 'CID-SMILES.parquet'))
        print('Done.')
    elif process_name == 'MassToDataFrame':
        print('Reading CID-Mass file...', end=' ')
        file_path = '/workspaces/Ms2z/mnt/data/smiles/pubchem/CID-Mass'
        mass_df = CidMassToDataFrame(file_path)
        print(f'Number of rows: {len(mass_df)}')
        print('Save the DataFrame as a Parquet file...', end=' ')
        mass_df.to_parquet(os.path.join(os.path.dirname(file_path), 'CID-Mass.parquet'))
        print('Done.')
    elif process_name == 'Merge':
        print('Merging CID files...', end=' ')
        file_path_list = [
            '/workspaces/Ms2z/mnt/data/smiles/pubchem/CID-SMILES.parquet',
            '/workspaces/Ms2z/mnt/data/smiles/pubchem/CID-Mass.parquet'
        ]
        column_name_mapping = {
            file_path_list[0]: ["CID", "SMILES"],
            file_path_list[1]: ["CID", "Formula", "MonoisotopicMass", "ExactMass"]
        }
        # column_type_mapping = {
        #     '/workspaces/Ms2z/mnt/data/mass/pubchem/CID-MASS': {
        #         "MonoisotopicMass": "float32",
        #         "ExactMass": "float32"
        #     }
        # }
        merged_df = merge_files_by_cid(file_path_list, column_name_mapping)
        merged_df.to_parquet('/workspaces/Ms2z/mnt/data/smiles/pubchem/CID-SMILES-Mass.parquet')
    elif process_name == 'Property':
        get_properties_from_cid(list(range(1,101)), num_workers=1)
