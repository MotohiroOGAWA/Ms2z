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
        
        # Internal cache for wide format data
        self._wide_format_cache = None
        self._last_processed_row = -1  # Tracks the last processed row
        self._total_rows = 0  # Cached total rows count

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

        self._total_rows += 1

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
        Load the TSV log file into a pandas DataFrame if necessary.

        Returns:
            pd.DataFrame: The data from the log file.
        """
        if self._total_rows > self._last_processed_row:
            # Load only if new rows exist
            data = pd.read_csv(self.log_file_path, delimiter='\t')
            self._last_processed_row = len(data)
            return data
        else:
            # No new rows, return None or empty DataFrame
            return None
    
    def to_wide_format(self):
        """
        Convert the TSV data to wide format with caching.

        Returns:
            dict: A dictionary with log types (e.g., 'train', 'validation') as keys
                  and corresponding DataFrames in wide format as values.
        """
        # Load data only if there are new rows
        new_data = self._load_log()

        if new_data is not None:
            if self._wide_format_cache is not None:
                # Update cache with new data
                new_wide_format = self._process_to_wide_format(new_data)
                for log_type, df in new_wide_format.items():
                    if log_type in self._wide_format_cache:
                        self._wide_format_cache[log_type] = pd.concat(
                            [self._wide_format_cache[log_type], df], ignore_index=True
                        )
                    else:
                        self._wide_format_cache[log_type] = df
            else:
                # First-time processing
                self._wide_format_cache = self._process_to_wide_format(new_data)

        return self._wide_format_cache
    
    def _process_to_wide_format(self, data):
        """
        Process the given data to wide format.

        Args:
            data (pd.DataFrame): DataFrame to process.

        Returns:
            dict: Wide format DataFrames by type.
        """
        data['target_name'] = data['target_name'].apply(lambda x: x.split(','))
        data['loss'] = data['loss'].apply(lambda x: list(map(float, x.split(','))))
        data['accuracy'] = data['accuracy'].apply(
            lambda x: [float(val) if val not in ['N/A', 'None'] else None for val in x.split(',')]
        )
        data['criterion'] = data['criterion'].apply(lambda x: x.split(','))

        unique_target_names = list(set(
            target for target_g in data['target_name'] for target in target_g
        ))

        grouped_data = {}
        for log_type, group in data.groupby('type'):
            transformed_data = group[['global_step', 'epoch', 'data_size', 'timestamp', 'learning_rate']].copy()
            for target in unique_target_names:
                transformed_data[f"{target}_loss"] = None
                transformed_data[f"{target}_accuracy"] = None

            for i, row in group.iterrows():
                for target, loss, accuracy in zip(row['target_name'], row['loss'], row['accuracy']):
                    transformed_data.loc[i, f"{target}_loss"] = loss
                    transformed_data.loc[i, f"{target}_accuracy"] = accuracy

            grouped_data[log_type] = transformed_data.reset_index(drop=True)

        return grouped_data

    def plot_loss_by_target(self, target_names=None, save_dir_name=None, xlim=None, ylim=None, legend_fontsize=10, separate_loss_and_accuracy=True):
        """
        Plot losses and accuracies for specified target_names. Optionally, separate or combine loss and accuracy.

        Args:
            target_names (list or None): List of target names or grouped target names (e.g., "targetA,targetB").
                                        If None, all target names are plotted individually.
            save_dir_name (str): Directory name where the plots will be saved.
            xlim (tuple): Tuple (xmin, xmax) for limiting the x-axis range (epochs).
            ylim (tuple): Tuple (ymin, ymax) for limiting the y-axis range (loss values).
            legend_fontsize (int, optional): Font size for the legend.
            separate_loss_and_accuracy (bool, optional): If True, plot loss and accuracy in separate plots. If False, combine them.
                                        Default is True.
        """
        # Convert the data to wide format
        wide_data_by_type = self.to_wide_format()

        # Automatically extract all unique targets if target_names is None
        if target_names is None:
            type_names = list(wide_data_by_type.keys())
            target_names = set(
                col[:-len('_loss')] for col in wide_data_by_type[type_names[0]].columns if col.endswith('_loss')
            )

        # Process each specified target name or grouped targets
        for target_group in target_names:
            targets = target_group.split(',')

            if separate_loss_and_accuracy:
                # Plot losses and accuracies in separate plots
                for target in targets:
                    plt.figure(figsize=(12, 8))
                    flag = False

                    for type_name, wide_data in wide_data_by_type.items():
                        loss_col = f"{target}_loss"

                        if loss_col in wide_data:
                            flag = True
                            plt.plot(wide_data['global_step'], wide_data[loss_col],
                                    label=f"{type_name} Loss ({target})", linestyle='-')

                    plt.title(f"Loss for {target}")
                    plt.xlabel('Global Step')
                    plt.ylabel('Loss Value')
                    plt.legend(fontsize=legend_fontsize)
                    plt.grid(True)

                    if xlim:
                        plt.xlim(xlim)
                    if ylim:
                        plt.ylim(ylim)

                    if flag:
                        if save_dir_name:
                            save_dir = os.path.join(os.path.dirname(self.log_file_path), save_dir_name)
                            os.makedirs(save_dir, exist_ok=True)
                            plt.savefig(f"{save_dir}/{target}_loss_plot.png")
                        else:
                            plt.show()

                    plt.close()

                    # Plot accuracy
                    plt.figure(figsize=(12, 8))
                    flag = False

                    for type_name, wide_data in wide_data_by_type.items():
                        accuracy_col = f"{target}_accuracy"

                        if accuracy_col in wide_data and wide_data[accuracy_col].notna().any():
                            flag = True
                            plt.plot(wide_data['global_step'], wide_data[accuracy_col],
                                    label=f"{type_name} Accuracy ({target})", linestyle='-')

                    plt.title(f"Accuracy for {target}")
                    plt.xlabel('Global Step')
                    plt.ylabel('Accuracy Value')
                    plt.legend(fontsize=legend_fontsize)
                    plt.grid(True)

                    if xlim:
                        plt.xlim(xlim)
                    if ylim:
                        plt.ylim(ylim)

                    if flag:
                        if save_dir_name:
                            plt.savefig(f"{save_dir}/{target}_accuracy_plot.png")
                        else:
                            plt.show()

                    plt.close()

            else:
                # Combine loss and accuracy in the same plot
                plt.figure(figsize=(12, 8))

                for target in targets:
                    for type_name, wide_data in wide_data_by_type.items():
                        loss_col = f"{target}_loss"
                        accuracy_col = f"{target}_accuracy"

                        if loss_col in wide_data:
                            plt.plot(wide_data['global_step'], wide_data[loss_col],
                                    label=f"{type_name} Loss ({target})", linestyle='-')

                        if accuracy_col in wide_data and wide_data[accuracy_col].notna().any():
                            plt.plot(wide_data['global_step'], wide_data[accuracy_col],
                                    label=f"{type_name} Accuracy ({target})", linestyle='--')

                plt.title(f"Metrics for {' & '.join(targets)}")
                plt.xlabel('Global Step')
                plt.ylabel('Value')
                plt.legend(fontsize=legend_fontsize)
                plt.grid(True)

                if xlim:
                    plt.xlim(xlim)
                if ylim:
                    plt.ylim(ylim)

                if save_dir_name:
                    save_dir = os.path.join(os.path.dirname(self.log_file_path), save_dir_name)
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(f"{save_dir}/{'_'.join(targets)}_metrics_plot.png")
                else:
                    plt.show()

                plt.close()
