import csv
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

class TSVLogger:
    def __init__(self, log_file_path, timestamp_format='%Y%m%d%H%M%S', extra_columns=None):
        """
        Initialize the TSVLogger class.
        
        Args:
            log_file_path (str): Base path to the log file where data will be saved in TSV format.
            timestamp_format (str): Format for the timestamp to be added to the file name. Default is '%Y%m%d%H%M%S'.
            extra_columns (list, optional): List of additional column names.
        """
        if timestamp_format:
            timestamp = datetime.now().strftime(timestamp_format)
            base, ext = os.path.splitext(log_file_path)
            log_file_path = f"{base}_{timestamp}{ext}"
        
        self.log_file_path = log_file_path
        self.extra_columns = extra_columns if extra_columns else []

        # Check if the log file exists, if not, create it and write the header
        if not os.path.exists(self.log_file_path):
            self._write_header()

    def _write_header(self):
        """
        Writes the header to the TSV file if it does not already exist.
        """
        with open(self.log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            self.header = [
                'global_step',  # Current global step
                'epoch',        # Current epoch
                'type',         # train, validation, test
                'data_size',    # Size of the data used in the current step
                'timestamp',    # Timestamp of the log
                'learning_rate',# Optional: Learning rate
                'target_name',  # List of target names
                'loss',         # List of losses
                'accuracy',     # List of accuracies
                'criterion'     # List of loss types
            ] + self.extra_columns
            writer.writerow(self.header)
    
    def log(self, log_type, epoch, global_step, data_size, target_data, learning_rate=None, timestamp=None, extra_columns=None):
        """
        Logs the specified data to the TSV file, with optional additional columns.

        Args:
            log_type (str): Type of log, e.g., 'train', 'validation', 'test'.
            epoch (int): Current epoch number.
            global_step (int): Current global step.
            data_size (int): Size of the data (e.g., batch size or total data points).
            target_data (dict): Dictionary containing target-specific data. Format:
                {
                    "target1": {"criterion": "cross_entropy", "loss": 0.4, "accuracy": 0.85},
                    "target2": {"criterion": "mse", "loss": 0.3, "accuracy": None}
                }
            learning_rate (float, optional): Learning rate. Default is None.
            timestamp (str, optional): Timestamp of the log. If None, the current time will be used.
            extra_columns (dict, optional): Additional columns to log. Format:
                {"column_name1": value1, "column_name2": value2}.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Extract target-specific data
        target_names = list(target_data.keys())
        loss_values = [target_data[target]['loss'] for target in target_names]
        accuracy_values = [target_data[target].get('accuracy', 'N/A') for target in target_names]
        criterions = [target_data[target]['criterion'] for target in target_names]

        # Prepare the base row
        row = {
            'global_step': global_step,
            'epoch': epoch,
            'type': log_type,
            'data_size': data_size,
            'timestamp': timestamp,
            'learning_rate': learning_rate if learning_rate is not None else 'N/A',
            'target_name': ','.join(target_names),
            'loss': ','.join(map(str, loss_values)),
            'accuracy': ','.join(map(str, accuracy_values)),
            'criterion': ','.join(criterions)
        }

        # Add extra columns if provided
        if extra_columns:
            row.update(extra_columns)

        # Ensure the row matches the header order
        with open(self.log_file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, delimiter='\t', fieldnames=self.header)
            writer.writerow(row)

    @staticmethod
    def load(file_path):
        """
        Load the TSV log file into a pandas DataFrame.
        
        Args:
            file_path (str): Path to the TSV log file.
        
        Returns:
            pd.DataFrame: Data from the TSV log file.
        """
        loggers = TSVLogger(file_path, timestamp_format=None)
        return loggers

    def _load_log(self):
        """
        Load the TSV log file into a pandas DataFrame.
        
        Returns:
            pd.DataFrame: Data from the TSV log file.
        """
        return pd.read_csv(self.log_file_path, delimiter='\t')
    
    def to_wide_format(self):
        """
        Transform the TSV data into a wide format where each target has its own columns,
        separated by `type` (e.g., 'train', 'validation').

        Returns:
            dict: A dictionary where keys are `type` values (e.g., 'train', 'validation'),
                and values are transformed DataFrames with target-specific columns.
        """
        # Load log data
        data = self._load_log()

        # Ensure target data columns are properly converted from string to list
        data['target_name'] = data['target_name'].apply(lambda x: x.split(','))
        data['loss'] = data['loss'].apply(lambda x: list(map(float, x.split(','))))
        data['accuracy'] = data['accuracy'].apply(
            lambda x: [
                float(val) if val not in ['N/A', 'None'] else None
                for val in x.split(',')
            ]
        )
        data['criterion'] = data['criterion'].apply(lambda x: x.split(','))

        # Extract unique target names
        unique_target_names = list(set(
            target for target_g in data['target_name'] for target in target_g
        ))

        # Group data by `type`
        grouped_data = {}
        for log_type, group in data.groupby('type'):
            # Initialize transformed DataFrame for this `type`
            transformed_data = group[['global_step', 'epoch', 'data_size', 'timestamp', 'learning_rate']].copy()

            # Add columns for each target's metrics
            for target in unique_target_names:
                transformed_data[f"{target}_loss"] = None
                transformed_data[f"{target}_accuracy"] = None

            # Populate the transformed DataFrame
            for i, row in group.iterrows():
                for target, loss, accuracy in zip(row['target_name'], row['loss'], row['accuracy']):
                    transformed_data.loc[i, f"{target}_loss"] = loss
                    transformed_data.loc[i, f"{target}_accuracy"] = accuracy

            grouped_data[log_type] = transformed_data.reset_index(drop=True)

        return grouped_data

    def plot_loss(self, target_names=None, save_dir_name=None, xlim=None, ylim=None, legend_fontsize=10):
        """
        Plot losses for specified target_names. If target_names is None, plot all targets individually.

        Args:
            target_names (list or None): List of target names or grouped target names (e.g., "targetA,targetB").
                                        If None, all target names are plotted individually.
            save_dir_name (str): Directory name where the plots will be saved.
            xlim (tuple): Tuple (xmin, xmax) for limiting the x-axis range (epochs).
            ylim (tuple): Tuple (ymin, ymax) for limiting the y-axis range (loss values).
            legend_fontsize (int, optional): Font size for the legend.
        """
        # Convert the data to wide format
        wide_data_by_type = self.to_wide_format()
        type_names = list(wide_data_by_type.keys())

        # Extract all unique targets if target_names is None
        if target_names is None:
            target_names = set(col[:-len('_loss')] for col in wide_data_by_type[type_names[0]].columns if col.endswith('_loss'))

        # Process each specified target or grouped targets
        for target_group in target_names:
            # Handle grouped targets
            targets = target_group.split(',')

            for type_name, wide_data in wide_data_by_type.items():
                plt.figure(figsize=(10, 6))
                for target in targets:
                    loss_col = f"{target}_loss"

                    if loss_col not in wide_data:
                        print(f"Warning: No data found for target '{target}'. Skipping.")
                        continue

                    # Plot the losses
                    plt.plot(wide_data['global_step'], wide_data[loss_col], label=f"{target} Loss")

                # Add plot details
                plt.title(f"{type_name} Loss for {' & '.join(targets)}")
                plt.xlabel('Global Step')
                plt.ylabel('Loss Value')
                plt.legend(fontsize=legend_fontsize)
                plt.grid(True)

                # Set x and y axis limits if provided
                if xlim:
                    plt.xlim(xlim)
                if ylim:
                    plt.ylim(ylim)

                # Save or show the plot
                if save_dir_name:
                    save_dir = os.path.join(os.path.dirname(self.log_file_path), save_dir_name)
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(f"{save_dir}/{'_'.join(targets)}_loss_plot.png")
                else:
                    plt.show()
            plt.close()