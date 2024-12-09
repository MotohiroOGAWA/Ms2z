import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import os

def plot_counter_distribution(counter: Counter, save_file=None, display_thresh=None, y_scale='linear', bin_width=None):
    """
    Plot a histogram and cumulative distribution based on a Counter object and return distribution data as a DataFrame.
    
    Parameters:
        counter (Counter): A Counter object where keys are categories or values and counts are their frequency.
        save_file (str, optional): Path to save the plot. If None, the plot is displayed.
        display_thresh (int, optional): Threshold to display bins with counts above this value.
        y_scale (str, optional): Y-axis scale ('linear' or 'log').
        bin_width (int, optional): Width of bins for the histogram.
    
    Returns:
        pandas.DataFrame: A DataFrame containing:
            - 'Bin Start': Start of each bin (used as index).
            - 'Count': Frequency in each bin.
            - 'Cumulative Count': Cumulative frequency.
            - 'Cumulative Percentage': Cumulative percentage.
    """
    # Ensure counter values are numeric
    values = np.array(list(counter.values()), dtype=np.float64)
    
    # Create bins if bin_width is specified
    if bin_width is not None:
        bins = np.arange(0, values.max() + bin_width, bin_width)
    else:
        bins = 'auto'  # Use automatic binning if bin_width is not specified

    # Compute histogram
    counts, bin_edges = np.histogram(values, bins=bins)

    # Apply threshold: Remove bins where the count is below the threshold
    if display_thresh is not None:
        valid_bins = counts >= display_thresh
        counts = counts[valid_bins]
        bin_edges = bin_edges[:-1][valid_bins]  # Use bin starts for valid bins only
    else:
        bin_edges = bin_edges[:-1]  # Use bin starts only

    # Calculate cumulative percentage
    cumulative_counts = np.cumsum(counts)
    cumulative_percentage = cumulative_counts / cumulative_counts[-1] * 100

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the histogram using the filtered counts and bins
    ax1.bar(bin_edges, counts, width=bin_width if bin_width else 1, align='edge', edgecolor='black', color='skyblue')
    ax1.set_xlabel('Value (Binned)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Frequency Distribution and Cumulative Count (Histogram with bin width = {bin_width})')

    # Set Y-axis scale
    if y_scale == 'log':
        ax1.set_yscale('log')

    # Set X-axis limits to ensure all valid data is visible
    if len(bin_edges) > 0:
        ax1.set_xlim(bin_edges.min(), bin_edges.max() + (bin_width if bin_width else 1))

    # Create a secondary axis for the cumulative percentage line plot
    ax2 = ax1.twinx()
    ax2.plot(bin_edges, cumulative_percentage, color='red', linestyle='-', linewidth=2)
    ax2.set_ylabel('Cumulative Percentage (%)')
    ax2.set_ylim(0, 100)

    # Save the plot to a file if save_file is specified
    if save_file:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        plt.savefig(save_file)
        print(f"Chart saved to {save_file}")
    else:
        # Display the chart
        plt.show()
    
    # Create a DataFrame with the distribution data
    df = pd.DataFrame({
        'Count': counts,
        'CumulativeCount': cumulative_counts,
        'CumulativePercentage': cumulative_percentage
    }, index=bin_edges)
    df.index.name = 'BinStart'

    return df
