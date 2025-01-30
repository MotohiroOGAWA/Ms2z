import pandas as pd
import matplotlib.pyplot as plt

def plot_precursor_type_distribution(df, precursor_type_column='PrecursorType', top=None, save_file=None, chart_type='bar', y_scale='linear', threshold=None):
    """
    Plots the distribution of the specified column (default is 'PrecursorType') in the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the specified precursor type column.
        precursor_type_column (str, optional): The column name to use for plotting. Default is 'PrecursorType'.
        top (int, optional): The number of top values to display. Default is None, which displays all.
        save_file (str, optional): The file path to save the chart. If None, the chart will be displayed.
        chart_type (str, optional): The type of chart to create ('bar' or 'pie'). Default is 'bar'.
        y_scale (str, optional): The scale for the Y-axis ('linear' or 'log'). Default is 'linear'.
        threshold (int, optional): The minimum count value to display. Default is None, which displays all.
    """
    # Check if the specified column exists in the DataFrame
    if precursor_type_column not in df.columns:
        raise ValueError(f"Column '{precursor_type_column}' not found in the DataFrame.")
    
    # Count the occurrences of each value in the specified column
    precursor_counts = df[precursor_type_column].value_counts()

    # Apply threshold if specified
    if threshold is not None:
        precursor_counts = precursor_counts[precursor_counts >= threshold]

    # If top is specified, limit the number of displayed items
    if top is not None:
        precursor_counts = precursor_counts.head(top)

    # Create the plot based on the specified chart type
    plt.figure(figsize=(10, 6))

    if chart_type == 'bar':
        # Bar chart
        precursor_counts.plot(kind='bar')
        plt.ylabel('Count')
        plt.title(f'Distribution of {precursor_type_column} (Bar Chart)')
        
        # Set Y-axis scale
        if y_scale == 'log':
            plt.yscale('log')
    elif chart_type == 'pie':
        # Pie chart
        precursor_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.ylabel('')
        plt.title(f'Distribution of {precursor_type_column} (Pie Chart)')
    else:
        raise ValueError("chart_type must be either 'bar' or 'pie'")

    # Save the plot to a file if save_file is specified
    if save_file:
        plt.savefig(save_file)
        print(f"Chart saved to {save_file}")
    else:
        # Display the chart
        plt.show()