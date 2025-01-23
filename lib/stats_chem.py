import pandas as pd
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import AddHs
from collections import defaultdict
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import os
import glob
import matplotlib.pyplot as plt

try:
    from ..lib.calc_formula import formula_to_dict
    from ..lib.chem_data import calc_exact_mass
except ImportError:
    from calc_formula import formula_to_dict
    from chem_data import calc_exact_mass

# Disable RDKit logging
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)  # Only show critical errors, suppress warnings and other messages

# Function to count atoms from a SMILES string, including hydrogen
def count_atoms_and_mass(smiles, no_ions):
    atom_counter = defaultdict(int)
    
    # Convert SMILES string to an RDKit Mol object
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        if no_ions:
            has_ions = contains_ions(mol)
            if has_ions:
                return None

        # Add explicit hydrogens
        mol = AddHs(mol)
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        element_counts = formula_to_dict(formula)
        exact_mass = calc_exact_mass(element_counts)
        s_mol = Chem.RemoveHs(mol)
        smiles = Chem.MolToSmiles(s_mol, canonical=True)

    except Exception as e:
        # Handle the error and print the error message
        # print(f"Error parsing SMILES: {smiles}, {e}")
        return None
    
    # Loop through all atoms in the molecule and count them, including hydrogens
    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        atom_counter[atom_symbol] += 1
    
    # Return the final atom count as a dictionary
    return smiles, dict(atom_counter), exact_mass

def contains_ions(mol):
    return any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())

# Function to process a chunk of SMILES and save the atom counts
def process_smiles_chunk(smiles_chunk, ids_chunk, output_dir, chunk_index, progress_queue, no_ions):
    data = []
    
    # For each SMILES, count the atoms and append to the data list
    for i, smiles in enumerate(smiles_chunk):
        result = count_atoms_and_mass(smiles, no_ions)
        if result is not None:
            smiles, atom_counts, exact_mass = result
            if ids_chunk is not None:
                atom_counts['ID'] = ids_chunk[i]
            atom_counts['SMILES'] = smiles
            atom_counts['ExactMass'] = exact_mass
            data.append(atom_counts)
    
    # Create a DataFrame from the atom count data
    df = pd.DataFrame(data)
    
    # Ensure SMILES column is first, and cast counts to integer type
    if 'ID' in df.columns:
        columns = ['ID', 'SMILES', 'ExactMass']
    else:
        columns = ['SMILES', 'ExactMass']
    columns = columns + [col for col in df.columns if col != 'ID' and col != 'SMILES' and col != 'ExactMass']
    df = df[columns].fillna(0).astype({col: int for col in df.columns if col != 'ID' and col != 'SMILES' and col != 'ExactMass'})  # fill NaN with 0 and convert counts to int
    
    # Save the DataFrame to a Parquet file
    output_file = os.path.join(output_dir, f"smiles_chunk_{chunk_index}.parquet")
    df.to_parquet(output_file)
    
    # Update the progress queue
    progress_queue.put(1)

# Function to process SMILES from a file using parallel processing with a progress bar
def count_atoms_in_file(smiles_file, output_dir, smiles_per_chunk=100, num_workers=None, no_ions=True):
    # Read the SMILES from the file into a list
    with open(smiles_file, 'r') as f:
        all_smiles = [s.strip() for line in f if line.strip() for s in line.strip().split(".")]
    
    count_atoms_in_file_from_list(all_smiles, output_dir, smiles_per_chunk=smiles_per_chunk, num_workers=num_workers, no_ions=no_ions)


def count_atoms_in_file_from_list(smiles_list, output_dir, smiles_per_chunk=100, num_workers=None, no_ions=True, id_list=None):
    assert id_list is None or len(id_list) == len(smiles_list), "ID list must have the same length as the SMILES list"

    if num_workers is None:
        num_workers = cpu_count()  # Set number of workers based on CPU cores

    # Create the output directory if it doesn't exist
    chunk_dir = os.path.join(output_dir, 'chunks')
    if not os.path.exists(chunk_dir):
        os.makedirs(chunk_dir)
    # Find all Parquet files in the directory
    chunk_files = glob.glob(os.path.join(chunk_dir, "*.parquet"))
    # Remove the individual Parquet files after merging
    for chunk_file in chunk_files:
        os.remove(chunk_file)
    final_output_parquet = os.path.join(output_dir, 'atom_count.parquet')
    distribution_output_parquet = os.path.join(output_dir, 'distribution_output.parquet')
    save_distribution_dir = os.path.join(output_dir, 'distribution_plots')

    # Split the SMILES list into chunks of size 'smiles_per_chunk'
    chunks = [smiles_list[i:i + smiles_per_chunk] for i in range(0, len(smiles_list), smiles_per_chunk)]
    if id_list is None:
        ids_chunks = [None]*len(chunks)
    else:
        ids_chunks = [id_list[i:i + smiles_per_chunk] for i in range(0, len(id_list), smiles_per_chunk)]

    
    # Use a multiprocessing Manager to track progress
    manager = Manager()
    progress_queue = manager.Queue()

    # Use a tqdm progress bar to track progress
    total_chunks = len(chunks)
    progress_bar = tqdm(total=total_chunks, desc="Processing SMILES chunks")

    # Function to update the progress bar
    def update_progress_bar(q):
        while not q.empty():
            q.get()
            progress_bar.update(1)

    # Use a multiprocessing pool to process chunks in parallel
    with Pool(num_workers) as pool:
        for chunk_index, smiles_chunk in enumerate(chunks):
            pool.apply_async(process_smiles_chunk, args=(smiles_chunk, ids_chunks[chunk_index], chunk_dir, chunk_index, progress_queue, no_ions))

        # Monitor progress and update the progress bar
        while progress_bar.n < total_chunks:
            update_progress_bar(progress_queue)

        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()

    # Close the progress bar
    progress_bar.close()

    # After processing, collect all the Parquet files and merge them
    print("Merging Parquet files...")
    count_atoms_df = merge_parquet_files(chunk_dir, final_output_parquet)

    print("All SMILES processing completed and saved to Parquet files.")

    distribution_df = calculate_atom_ratios(count_atoms_df)
    distribution_df.to_parquet(distribution_output_parquet)

    # Plot and save the ratios bar chart
    ratios_plot_path = plot_and_save_ratios_bar_chart(distribution_df, save_path=os.path.join(output_dir, 'ratios_bar_chart.png'))
    print(f"Ratios bar chart saved to {ratios_plot_path}")
    plot_atom_count_distribution(distribution_df, threshold=1, save_dir=save_distribution_dir, log_scale=True)

# Function to merge all Parquet files into a single Parquet file and delete the smaller ones
def merge_parquet_files(input_dir, output_file):
    # Find all Parquet files in the directory
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    
    # Read and concatenate all Parquet files into a single DataFrame
    dataframes = [pd.read_parquet(file) for file in parquet_files]

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = combined_df.fillna(0)
    combined_df = combined_df.astype({col: int for col in combined_df.columns if col != 'ID' and col != 'SMILES' and col != 'ExactMass'})  # fill NaN with 0 and convert counts to int
    # combined_df = combined_df.drop_duplicates(subset=['SMILES'])
    if 'ID' in combined_df.columns:
        combined_df = combined_df.sort_values('ID').sort_values(by='ID', key=lambda col: col.astype(int))

    # Save the combined DataFrame to a single Parquet file
    combined_df.to_parquet(output_file)
    print(f"Combined Parquet file saved to {output_file}")
    
    # Remove the individual Parquet files after merging
    for parquet_file in parquet_files:
        os.remove(parquet_file)
        print(f"Deleted {parquet_file}")

    try:
        os.rmdir(input_dir)
    except OSError:
        pass
    except Exception as e:
        pass
    
    return combined_df

# Function to calculate the ratios of compounds containing each atom
def calculate_atom_ratios(df):
    total_compounds = len(df)
    ratios = {}
    max_count = {}
    for c in tqdm(df.columns, desc='Calculating atom ratios'):
        if c == 'ID' or c == 'SMILES' or c == 'ExactMass':
            continue
        ratios[c] = (df[c] > 0).sum() / total_compounds
        max_count[c] = df[c].max()

    all_max_count = max(max_count.values())

    bin_edges = [i for i in range(0, all_max_count+1)]
    distributions = {}
    for c in tqdm(df.columns, desc='Calculating atom distribution'):
        if c == 'ID' or c == 'SMILES' or c == 'ExactMass':
            continue
        distribution = np.histogram(df[c], bins=bin_edges)
        distributions[c] = distribution[0]
    
    distribution_df = pd.DataFrame(distributions, index=[f"{bin_edges[i]}" for i in range(len(bin_edges)-1)])
    distribution_df = pd.concat([pd.DataFrame(ratios, index=['ratios']), distribution_df])
    
    return distribution_df

# Function to plot and save the ratios bar chart, and display if no save_path is provided
def plot_and_save_ratios_bar_chart(distribution_df, save_path=None):
    # Get ratios data and sort it in descending order
    ratios_df = distribution_df.loc['ratios']
    sorted_ratios = ratios_df.T.sort_values(ascending=False)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_ratios.index, sorted_ratios.values)
    
    # Adjust height for even-indexed bars to be slightly lower
    for i, bar in enumerate(bars):
        if i % 2 == 0:
            bar.set_height(bar.get_height() * 0.9)  # Lower even-indexed bars by 10%
    
    # Add titles and labels
    plt.title('Ratios of Atoms (Descending)')
    plt.xlabel('Atom')
    plt.ylabel('Ratio')
    
    # Adjust X-axis labels manually for even indexes to shift them down
    x_labels = sorted_ratios.index
    ax = plt.gca()
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    
    # Iterate through labels and shift even labels down
    for i, label in enumerate(ax.get_xticklabels()):
        if i % 2 == 0:
            label.set_y(label.get_position()[1] - 0.05)  # Shift even labels slightly down
    
    # Adjust layout for better fit
    plt.tight_layout()
    
    # Save the figure if a path is provided, otherwise display the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        return save_path
    else:
        plt.show()
    
    return sorted_ratios

def calculate_tick_interval(data_min, data_max, num_ticks, integer_only=False):
    """Calculate the tick interval using the 1, 2, 5, 10, 20, 50, 100... rule."""
    range_size = data_max - data_min
    
    # Avoid division by zero if data_min and data_max are the same
    if range_size == 0:
        return 1  # Default interval if no range is available
    
    raw_interval = range_size / num_ticks
    
    # Determine the appropriate "nice" interval
    scale = 10 ** np.floor(np.log10(raw_interval))  # Find the scale (1, 10, 100, etc.)
    nice_intervals = np.array([1, 2, 5])  # Possible multipliers
    
    # Ensure the index stays within bounds
    index = np.searchsorted(nice_intervals, raw_interval / scale)
    if index >= len(nice_intervals):
        index = len(nice_intervals) - 1  # Use the largest multiplier if out of bounds
    
    interval = nice_intervals[index] * scale

    # If integer_only is True, round up to the nearest integer
    if integer_only:
        interval = np.ceil(interval)
    
    return max(1, int(interval))  # Ensure interval is never zero

def plot_atom_count_distribution(distribution_df, bin_width=1, atom_columns=None, save_dir=None, num_xticks=10, log_scale=False, threshold=None, cumulative_percentage_line=None):
    """
    Plot the distribution of atom counts within molecules.

    Args:
        distribution_df (pd.DataFrame): DataFrame containing the distribution of atom counts.
        bin_width (int): The width of the bins used for histogram creation.
        atom_columns (list): List of columns representing atom counts to be plotted.
        save_dir (str): Directory to save the plots. If None, plots are shown instead of saved.
        num_xticks (int): Number of X-axis tick marks to display.
        log_scale (bool): Whether to use a logarithmic Y-axis scale.
        threshold (int): Minimum count value for bins to be displayed. Bins with counts below this threshold will be aggregated into "Other".
        cumulative_percentage_line (float): Draw a vertical line at the bin where the specified cumulative percentage is reached (e.g., 0.95 for 95%).
    """
    distribution_paths = []
    
    # If atom_columns are not provided, use all columns from the dataframe
    if atom_columns is None:
        atom_columns = [c for c in distribution_df.columns if c != 'ratios']
    
    # Exclude 'ratios' from the distribution data
    distribution_df = distribution_df.loc[distribution_df.index != 'ratios', atom_columns]

    # Extract bin edges for the histogram based on the index (which is already binned)
    bins = [int(i) for i in distribution_df.index]
    bin_edges = [i for i in range(0, max(bins)+1, bin_width)]
    if bin_edges[-1] < max(bins):
        bin_edges.append(max(bins))
    
    # Initialize tqdm progress bar
    with tqdm(total=len(atom_columns), desc="Plotting atom count distributions") as pbar:
        for atom in atom_columns:
            fig, ax1 = plt.subplots(figsize=(8, 5))

            # Sum the counts for each bin_width (combining bins within the bin width)
            atom_distribution = np.add.reduceat(distribution_df[atom].values, np.arange(0, len(bins), bin_width))

            # Apply threshold: combine bins under the threshold into the last "Other" bin
            if threshold is not None:
                low_value_bins = atom_distribution < threshold
                if np.any(low_value_bins):
                    other_sum = np.sum(atom_distribution[low_value_bins])
                    atom_distribution = np.append(atom_distribution[~low_value_bins], other_sum)
                    xtick_labels = [f'{bin_edges[i * bin_width]}' for i in range(len(atom_distribution) - 1)]
                    xtick_labels.append('Other')
                else:
                    xtick_labels = [f'{bin_edges[i * bin_width]}' for i in range(len(atom_distribution))]
            else:
                xtick_labels = [f'{bin_edges[i * bin_width]}' for i in range(len(atom_distribution))]

            # Create a bar chart using the summed data for each bin
            ax1.bar(range(len(atom_distribution)), atom_distribution, width=1.0)
            
            # Add title and labels
            ax1.set_title(f'Atom Count Distribution in {atom}')
            ax1.set_xlabel('Bins')
            ax1.set_ylabel('Frequency')

            # Determine appropriate xtick interval based on the range and desired number of ticks
            data_min = 0
            data_max = len(atom_distribution)
            tick_interval = calculate_tick_interval(data_min, data_max, num_xticks)
            
            # Generate xtick positions and set them along with the calculated labels
            xtick_positions = np.arange(0, len(atom_distribution), tick_interval)
            
            # Ensure the "Other" label is at the last position
            if len(xtick_positions) == 0 or xtick_positions[-1] < len(atom_distribution) - 1:
                xtick_positions = np.append(xtick_positions, len(atom_distribution) - 1)
            
            # Adjust the xtick labels based on the tick interval
            xtick_labels = [f'{int(bin_edges[i * bin_width])}' for i in xtick_positions[:-1]]
            xtick_labels.append('Other')  # Ensure "Other" is the last label
            
            ax1.set_xticks(xtick_positions)
            ax1.set_xticklabels(xtick_labels)  # No rotation
            
            # Apply log scale if requested
            if log_scale:
                ax1.set_yscale('log')

            # Cumulative sum for the stacked line plot
            cumulative_values = np.cumsum(atom_distribution)
            
            # Create secondary axis for percentage (second y-axis)
            ax2 = ax1.twinx()
            ax2.plot(range(len(cumulative_values)), cumulative_values / cumulative_values[-1] * 100, color='red', linestyle='-')  # Solid line only
            ax2.set_ylabel('Cumulative Percentage (%)')

            # Add cumulative percentage line if requested
            if cumulative_percentage_line is not None:
                cumulative_percentage = cumulative_values / cumulative_values[-1] * 100
                line_idx = np.argmax(cumulative_percentage >= cumulative_percentage_line * 100)
                ax1.axvline(x=line_idx, color='green', linestyle='--', label=f'{cumulative_percentage_line*100}% Line')
                ax1.legend()
                print(f'{cumulative_percentage_line} Line: {bin_edges[line_idx]}')

            # Adjust layout for better fit
            fig.tight_layout()

            # Save the figure if a path is provided, otherwise display the plot
            if save_dir:
                path = f'{save_dir}/{atom}_atom_count_distribution.png'
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(path)
                plt.close()
                distribution_paths.append(path)
            else:
                plt.show()

            # Update tqdm progress bar
            pbar.update(1)

    return distribution_paths if save_dir else None


def plot_exact_mass_distribution(atom_counts_df, exact_mass_column='ExactMass', top=None, save_file=None, y_scale='linear', threshold=None, bin_width=None, cumulative_percentage_line=None):
    """
    Plots the distribution of the exact mass in the given DataFrame with specified bin width for histogram,
    hiding bins with counts below the threshold. Also adds a cumulative line plot on a secondary y-axis,
    and allows specifying a cumulative percentage line (e.g., 0.95 for 95%).

    Args:
        atom_counts_df (pd.DataFrame): The input DataFrame containing the exact mass data.
        exact_mass_column (str, optional): The column name to use for exact mass. Default is 'ExactMass'.
        top (int, optional): The number of top values to display. Default is None, which displays all.
        save_file (str, optional): The file path to save the chart. If None, the chart will be displayed.
        y_scale (str, optional): The scale for the Y-axis ('linear' or 'log'). Default is 'linear'.
        threshold (int, optional): The minimum count value for bins to display. Default is None, which displays all.
        bin_width (float, optional): The width of each bin for histogram. Default is None, which calculates automatically.
        cumulative_percentage_line (float, optional): A value between 0 and 1 to draw a vertical line for the corresponding cumulative percentage.
    """
    # Check if the specified column exists in the DataFrame
    if exact_mass_column not in atom_counts_df.columns:
        raise ValueError(f"Column '{exact_mass_column}' not found in the DataFrame.")

    # Extract the data from the specified column
    exact_mass_data = atom_counts_df[exact_mass_column]

    # Apply top limit if specified
    if top is not None:
        exact_mass_data = exact_mass_data.nlargest(top)

    # Create bins if bin_width is specified
    if bin_width is not None:
        min_mass = exact_mass_data.min()
        max_mass = exact_mass_data.max()
        bins = np.arange(min_mass, max_mass + bin_width, bin_width)
    else:
        bins = 'auto'  # Use automatic binning if bin_width is not specified

    # Compute histogram
    counts, bin_edges = np.histogram(exact_mass_data, bins=bins)

    # Apply threshold: Remove bins where the count is below the threshold
    if threshold is not None:
        valid_bins = counts >= threshold
        counts = counts[valid_bins]
        bin_edges = bin_edges[:-1][valid_bins]  # Remove the last edge which is not a bin start

    # Calculate cumulative percentage
    cumulative_counts = np.cumsum(counts)
    cumulative_percentage = cumulative_counts / cumulative_counts[-1] * 100

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the histogram using the filtered counts and bins
    ax1.bar(bin_edges, counts, width=bin_width, align='edge', edgecolor='black', color='skyblue')
    ax1.set_xlabel('Exact Mass')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Distribution of {exact_mass_column} (Histogram with bin width = {bin_width})')

    # Set Y-axis scale
    if y_scale == 'log':
        ax1.set_yscale('log')

    # Set X-axis limits to ensure all valid data is visible
    if len(bin_edges) > 0:
        ax1.set_xlim(bin_edges.min(), bin_edges.max() + bin_width)

    # Create a secondary axis for the cumulative percentage line plot
    ax2 = ax1.twinx()
    ax2.plot(bin_edges, cumulative_percentage, color='red', linestyle='-', linewidth=2)
    ax2.set_ylabel('Cumulative Percentage (%)')
    ax2.set_ylim(0, 100)

    # Add a vertical line for the specified cumulative percentage
    if cumulative_percentage_line is not None:
        if not (0 <= cumulative_percentage_line <= 1):
            raise ValueError("cumulative_percentage_line must be between 0 and 1.")

        target_percentage = cumulative_percentage_line * 100
        target_index = np.searchsorted(cumulative_percentage, target_percentage)

        if target_index < len(bin_edges):
            target_mass = bin_edges[target_index]
            ax1.axvline(x=target_mass, color='green', linestyle='--', label=f'{target_percentage:.1f}% at {target_mass:.2f}')
            ax1.legend()
            print(f'{cumulative_percentage_line} Mass Line: {target_mass}')

    # Save the plot to a file if save_file is specified
    if save_file:
        plt.savefig(save_file)
        print(f"Chart saved to {save_file}")
    else:
        # Display the chart
        plt.show()

# Example usage:
# atom_counts_df = pd.DataFrame({'ExactMass': [10.5, 12.1, 10.5, 14.2, 12.1, 12.1, 15.3, 16.7, 15.9, 14.2]})
# plot_exact_mass_distribution(atom_counts_df, exact_mass_column='ExactMass', top=None, save_file=None, y_scale='linear', threshold=2, bin_width=0.5)

# Example usage
if __name__ == "__main__":
    smiles_file = "/workspaces/hgraph/mnt/ms2z/data/smiles/pubchem-canonical/unique_smiles.txt"
    # smiles_file = "/workspaces/hgraph/mnt/ms2z/data/smiles/pubchem-canonical/test.smi"
    output_dir = "/workspaces/hgraph/mnt/ms2z/data/smiles/pubchem-canonical/atom_counts"
    smiles_per_chunk = 10000
    num_workers = 50
    count_atoms_in_file(smiles_file, output_dir, smiles_per_chunk=smiles_per_chunk, num_workers=num_workers, no_ions=True)
