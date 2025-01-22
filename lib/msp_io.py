import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import warnings

import re
import dill
    

def read_msp_file(filepath, numpy_type=False, encoding='utf-8', save_file=None, overwrite=True, is_join_classyfire=False) -> pd.DataFrame:
    file_size = os.path.getsize(filepath)
    processed_size = 0
    line_count = 1

    cols = {} # Create a data list for each column
    cc = {} # Dictionary to convert column names
    # peaks = []
    peak = []
    max_peak_cnt = 0
    record_cnt = 1
    text = ""
    error_text = ""
    error_flag = False
    
    with open(filepath, 'r', encoding=encoding) as f:
        peak_flag = False
        with tqdm(total=file_size, desc="Read msp file") as pbar:
            for line in f.readlines():
                try:
                    if not peak_flag and line == '\n':
                        continue

                    text += line

                    if peak_flag and line == '\n':
                        peak_flag = False

                        if not error_flag:
                            #　エラーが生じなかった場合はデータを保存する
                            # peaks.append(np.array(peak))
                            if "Peak" not in cols:
                                cc["Peak"] = "Peak"
                                cols["Peak"] = [""] * record_cnt
                            cols["Peak"][-1] = ";".join([f"{mz},{intensity}" for mz, intensity in peak])
                            max_peak_cnt = max(max_peak_cnt, len(peak))
                        else:
                            # エラーが生じた場合はエラーデータとして保存する
                            error_text += f"Record: {record_cnt}\n" + f"Rows: {line_count}\n"
                            error_text += text + '\n\n'
                            error_flag = False
                            for k in cols:
                                if len(cols[k]) == record_cnt:
                                    cols[k].pop()
                                elif len(cols[k]) > record_cnt:
                                    error_text += f"Error: '{k}' has more data than the record count.\n"
                        text = ""
                        peak = []
                        record_cnt += 1
                        for k in cols:
                            cols[k].append("")
                    elif peak_flag:
                        # Handling cases where peaks are tab-separated or space-separated
                        if len(line.strip().split('\t')) == 2:
                            mz, intensity = line.strip().split('\t')
                        elif len(line.strip().split(' ')) == 2:
                            mz, intensity = line.strip().split(' ')
                        else:
                            raise ValueError(f"Error: '{line.strip()}' was not split correctly.")
                        mz, intensity = float(mz), float(intensity)
                        peak.append([mz, intensity])
                    else:
                        k, v = line.split(':', maxsplit=1)
                        k, v = k.strip(), v.strip()
                        k = k.replace(' ', '')
                        if k not in cc:
                            cc[k] = convert_to_unique_column(k)
                            if cc[k] == "CollisionEnergy":
                                cols[cc[k]+"1"] = [""] * record_cnt
                                cols[cc[k]+"2"] = [""] * record_cnt
                            else:
                                cols[cc[k]] = [""] * record_cnt
                        if cc[k] == "CollisionEnergy":
                            v1, v2 = convert_str_to_collision_energy(v)
                            cols[cc[k]+"1"][-1] = v1
                            cols[cc[k]+"2"][-1] = v2
                        elif cc[k] == "PrecursorType":
                            v = v.strip().replace(" ", "")
                            cols[cc[k]][-1] = to_precursor_type.get(v, v)
                        elif cc[k] == "Comments":
                            # Extract computed SMILES from comments
                            pattern = r'"computed SMILES=([^"]+)"'
                            match = re.search(pattern, v)
                            if match:
                                if "SMILES" not in cols:
                                    cc["SMILES"] = "SMILES"
                                    cols["SMILES"] = [""] * record_cnt
                                cols["SMILES"][-1] = match.group(1)
                        else:
                            cols[cc[k]][-1] = v
                        if k == "NumPeaks":
                            peak_flag = True
                    
                    line_count += 1
                    processed_size = len(line.encode(encoding)) + 1
                    pbar.update(processed_size)
                except Exception as e:
                    text = 'Error: ' + str(e) + '\n' + text
                    error_flag = True
                    pass

        # Append last peak data if file doesn't end with a blank line
        if line != '\n':
            cols["Peak"] = ";".join([f"{mz},{intensity}" for mz, intensity in peak])
            # peaks.append(np.array(peak))
            max_peak_cnt = max(max_peak_cnt, len(peak))

        # Remove last empty rows in metadata
        for k in cols:
            if cols[k][-1] != "":
                break
        else:
            for k in cols:
                del cols[k][-1]
        df = pd.DataFrame(data=cols, columns=cols.keys())

        # Convert data types according to the predefined types
        for c in df.columns:
            if c in msp_column_types:
                if msp_column_types[c] != "str":
                    df[c] = pd.to_numeric(df[c], errors='coerce').astype(msp_column_types[c])
        

    # # Warn if the number of peaks doesn't match the number of metadata entries
    # if len(peaks) != len(df):
    #     warnings.warn(f"Number of peaks ({len(peaks)}) and metadata ({len(df)}) do not match.")

    # df["Peak"] = peaks
    df['IdxOri'] = df.index

    if save_file is not None:
        try:
            save_msp_data(df, save_file, overwrite=overwrite)
        except FileExistsError as e:
            warnings.warn(str(e))

    if error_text != '':
        from datetime import datetime
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        with open(os.path.splitext(filepath)[0] + f"_error_{now}.txt", "w") as f:
            f.write(error_text)
            
    return df

def extract_peak(msp_df, idx):
    """
    Extracts peak data from the given DataFrame for specified indices.

    Parameters:
        msp_df (pd.DataFrame): DataFrame containing the "Peak" column with peak data.
        idx (int or list of int): Single index or a list of indices to extract peak data.

    Returns:
        np.ndarray or list of np.ndarray: If `idx` is a single index, returns a single 2D NumPy array.
                                           If `idx` is a list, returns a list of 2D NumPy arrays.
    """
    if isinstance(idx, (int, np.integer)):  # Single index
        peak_str = msp_df.loc[idx, "Peak"]
        peak = np.array([[float(mz), float(intensity)] for mz, intensity in [p.split(",") for p in peak_str.split(";")]])
        return peak
    elif isinstance(idx, (list, np.ndarray)):  # Multiple indices
        peaks = []
        for i in idx:
            peak_str = msp_df.loc[i, "Peak"]
            peak = np.array([[float(mz), float(intensity)] for mz, intensity in [p.split(",") for p in peak_str.split(";")]])
            peaks.append(peak)
        return peaks
    else:
        raise ValueError("idx must be an int or a list/array of int.")

def save_msp_data(peaks, metadata_df, save_dir, overwrite=True):
    # Check if the directory already exists and handle overwrite option
    if not overwrite and os.path.exists(save_dir):
        raise FileExistsError(f"Directory '{save_dir}' already exists. Set overwrite=True to overwrite the directory.")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save peaks and metadata
    dill.dump(peaks, open(os.path.join(save_dir, "peaks.pkl"), "wb"))
    metadata_df.to_parquet(os.path.join(save_dir, "meta.parquet"))

def load_msp_data(load_dir):
    peaks = dill.load(open(os.path.join(load_dir, "peaks.pkl"), "rb"))
    metadata_df = pd.read_parquet(os.path.join(load_dir, "meta.parquet"))
    return peaks, metadata_df 
    


    
# Padding 3D arrays with NaN values if the length of subarrays is inconsistent
def pad_3d_array_with_nan(arrays, max_rows=None):
    # If max_rows is None, calculate the maximum row size from subarrays
    if max_rows is None:
        max_rows = max(len(subarray) for subarray in arrays)
    
    # Initialize a new 3D array with NaN
    padded_array = np.full((len(arrays), max_rows, 2), np.nan)
    
    # Copy original array values into the new array
    for i, subarray in enumerate(arrays):
        for j, row in enumerate(subarray):
            if j < max_rows:
                padded_array[i, j, :len(row)] = row
    
    return padded_array


def pad_3d_array_with_nan(arrays, max_rows=None):
    # If max_rows is None, calculate the maximum row size from subarrays
    if max_rows is None:
        max_rows = max(subarray.shape[0] for subarray in arrays)
    
    # Initialize a new 3D array with NaN
    padded_array = np.full((len(arrays), max_rows, 2), np.nan)
    
    # Use tqdm to show progress while padding arrays
    for i, subarray in enumerate(tqdm(arrays, desc="Padding arrays")):
        for j in range(min(subarray.shape[0], max_rows)):
            padded_array[i, j, :] = subarray[j]
    
    return padded_array


# MSP column mappings for parsing and column naming
msp_column_names = {
    "Name": ["NAME"],
    "InChIKey": ["INCHIKEY"],
    "PrecursorMZ": ["Precursor_m/z", "PRECURSORMZ"],
    "PrecursorType": ["Precursor_type", "PRECURSORTYPE"],
    "SpectrumType": ["Spectrum_type", "SPECTRUMTYPE"],
    "InstrumentType": ["Instrument_type", "INSTRUMENTTYPE"],
    "Instrument": ["INSTRUMENT"],
    "IonMode": ["Ion_mode", "IONMODE"],
    "CollisionEnergy": ["Collision_energy", "COLLISIONENERGY"],
    "ExactMass": ["EXACTMASS"],
}

# Column data types for processing
msp_column_types = {
    "Name": "str",
    "InChIKey": "str",
    "PrecursorMZ": "Float32",
    "PrecursorType": "str",
    "SpectrumType": "str",
    "InstrumentType": "str",
    "Instrument": "str",
    "IonMode": "str",
    "CollisionEnergy": "str",
    "ExactMass": "Float32",
}

# Convert given column names to unique column names based on msp_column_names mapping
def convert_to_unique_columns(columns):
    unique_columns = []
    for c in columns:
        for msp_column, candidates in msp_column_names.items():
            if c in candidates:
                unique_columns.append(msp_column)
                break
        else:
            unique_columns.append(c)
    return unique_columns

# Convert a given column name to a unique column name based on msp_column_names mapping
def convert_to_unique_column(column):
    unique_column = ""
    for msp_column, candidates in msp_column_names.items():
        if column in candidates:
            unique_column = msp_column
            break
    else:
        unique_column = column
    return unique_column

# Convert columns to their predefined data types (strings if not specified)
def convert_to_types_str(columns):
    column_types = {}
    for c in columns:
        if c in msp_column_types:
            column_types[c] = msp_column_types[c]
        else:
            column_types[c] = "str"
    return column_types

# Remove decimals from strings for collision energy conversion
removed_decimals = []
def convert_str_to_collision_energy(text: str):
    removed_decimal = remove_decimal_numbers(text)
    if removed_decimal not in removed_decimals:
        removed_decimals.append(removed_decimal)

    # Extract numeric values for collision energy in different formats
    pattern = re.compile(r"([\d.]+)")
    match = pattern.search(text.replace(" ", ""))
    if match:
        return match.group(1), match.group(1)

    pattern = re.compile(r"([\d.]+)eV")
    match = pattern.search(text.replace(" ", ""))
    if match:
        return match.group(1), match.group(1)
    
    pattern = re.compile(r"([\d.]+)V")
    match = pattern.search(text.replace(" ", ""))
    if match:
        return match.group(1), match.group(1)
    
    pattern = re.compile(r"([\d.]+)HCD")
    match = pattern.search(text.replace(" ", ""))
    if match:
        return match.group(1), match.group(1)
    
    if text.replace(" ", "").startswith("Ramp"):
        pass

    pattern = re.compile(r"Ramp([\d.]+)\-([\d.]+)V")
    match = pattern.search(text.replace(" ", ""))
    if match:
        return match.group(1), match.group(2)
    
    return text, text

# Remove decimal numbers from a given text
def remove_decimal_numbers(text):
    # Regular expression pattern for decimal numbers
    pattern = r'\d+\.\d+|\d+\.|\.\d+|\d+'
    
    # Replace the decimal numbers with empty string
    result = re.sub(pattern, r'\\d', text.replace(" ", ""))
    
    return result


# PrecursorType column data mapping 
precursor_type_data = {
    "[M]+" : ["M", "[M]"],
    "[M+H]+": ["M+H", "[M+H]"],
    "[M-H]-": ["M-H", "[M-H]"],
    "[M+Na]+": ["M+Na", "[M+Na]"],
    "[M+K]+": ["M+K", "[M+K]"],
    "[M+NH4]+": ["M+NH4", "[M+NH4]"],
    }
to_precursor_type = {}
for precursor_type, data in precursor_type_data.items():
    to_precursor_type[precursor_type] = precursor_type
    for aliases in data:
        to_precursor_type[aliases] = precursor_type


if __name__ == "__main__":

    # Example 3D array with inconsistent sizes for demonstration
    peaks = [
        np.array([[1, 2], [3, 4], [5, 6]]),
        np.array([[1, 2], [3, 4]]),
        np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    ]

    # Without specifying max_rows (default is None)
    padded_peaks_with_nan_default = pad_3d_array_with_nan(peaks)
    print("Default max_rows (calculated):")
    print(padded_peaks_with_nan_default)

    # Specifying max_rows manually
    max_rows = 5
    padded_peaks_with_nan_specified = pad_3d_array_with_nan(peaks, max_rows)
    print("\nSpecified max_rows:")
    print(padded_peaks_with_nan_specified)
