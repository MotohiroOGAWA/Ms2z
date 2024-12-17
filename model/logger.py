import csv
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

class TSVLogger:
    def __init__(self, log_file_path):
        """
        Initialize the TSVLogger class.
        
        Args:
            log_file_path (str): Path to the log file where data will be saved in TSV format.
        """
        self.log_file_path = log_file_path
        
        # Check if the log file exists, if not, create it and write the header
        if not os.path.exists(self.log_file_path):
            self._write_header()

    def _write_header(self):
        """
        Writes the header to the TSV file if it does not already exist.
        """
        with open(self.log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            header = [
                'type',           # train, validation, test
                'epoch',          # Current epoch
                'global_step',    # Current global step
                'loss_type',      # Type of the loss
                'target_name',    # Name of the target for which loss is calculated
                'loss_value',     # Value of the loss
                'timestamp',      # Timestamp of the log
                'data_size',      # Size of the data used in the current step
                'accuracy',       # Optional: Accuracy or other metrics
                'learning_rate'   # Optional: Learning rate
            ]
            writer.writerow(header)
    
    def log(self, log_type, epoch, global_step, loss_type, target_name, loss_value, data_size, accuracy=None, learning_rate=None, timestamp=None):
        """
        Logs the specified data to the TSV file.
        
        Args:
            log_type (str): Type of log, e.g., 'train', 'validation', 'test'.
            epoch (int): Current epoch number.
            global_step (int): Current global step.
            loss_type (str): Type of loss, e.g., 'cross_entropy', 'mse'.
            target_name (str): Name of the target, e.g., 'classification', 'regression'.
            loss_value (float): Value of the loss.
            data_size (int): Size of the data (e.g., batch size or total data points).
            accuracy (float, optional): Accuracy or other performance metrics. Default is None.
            learning_rate (float, optional): Learning rate. Default is None.
            timestamp (str, optional): Timestamp of the log. If None, the current time will be used.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        log_data = [
            log_type,
            epoch,
            global_step,
            loss_type,
            target_name,
            loss_value,
            timestamp,
            data_size,
            accuracy if accuracy is not None else 'N/A',
            learning_rate if learning_rate is not None else 'N/A'
        ]
        
        with open(self.log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(log_data)


    def _load_log(self):
        """
        Load the TSV log file into a pandas DataFrame.
        
        Returns:
            pd.DataFrame: Data from the TSV log file.
        """
        return pd.read_csv(self.log_file_path, delimiter='\t')

    def plot_losses_by_target(self, save_dir_name="log_plot", xlim=None, ylim=None, legend_fontsize=None):
        """
        Plot losses for each target_name, separating train and validation data, and save the plots.

        Args:
            save_dir_name (str): Directory name where the plots will be saved.
            xlim (tuple): Tuple (xmin, xmax) for limiting the x-axis range (epochs).
            ylim (tuple): Tuple (ymin, ymax) for limiting the y-axis range (loss values).
            legend_fontsize (int, optional): Font size for the legend.
        """
        # Load log data into a DataFrame
        data = self._load_log()
        
        # Extract unique target_names
        target_names = data['target_name'].unique()

        # Plot for each target_name
        for target in target_names:
            target_data = data[data['target_name'] == target]
            train_data = target_data[target_data['type'] == 'train']
            val_data = target_data[target_data['type'] == 'validation']
            
            # Create a float epoch for smoother plotting (using global_step to approximate within an epoch)
            train_data['float_epoch'] = train_data['epoch'] + train_data['global_step'] / (train_data['global_step'].max() + 1)
            val_data['float_epoch'] = val_data['epoch'] + val_data['global_step'] / (val_data['global_step'].max() + 1)

            plt.figure(figsize=(10, 6))
            plt.plot(train_data['float_epoch'], train_data['loss_value'], label='Train Loss', color='blue')
            plt.plot(val_data['float_epoch'], val_data['loss_value'], label='Validation Loss', color='orange')
            plt.title(f'Loss for {target}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Value')
            plt.legend(fontsize=legend_fontsize)  # Set legend fontsize
            plt.grid(True)

            # Set x and y axis limits if provided
            if xlim:
                plt.xlim(xlim)
            if ylim:
                plt.ylim(ylim)
            
            # Save the plot to the specified directory
            save_dir = os.path.dirname(self.log_file_path) + f"/{save_dir_name}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(f"{save_dir}/{target}_loss_plot.png")
            plt.close()

    def plot_all_losses(self, save_dir_name="log_plot", xlim=None, ylim=None, legend_fontsize=None):
        """
        Plot all train and validation losses on the same graph, and save the plot.

        Args:
            save_dir_name (str): Directory name where the plots will be saved.
            xlim (tuple): Tuple (xmin, xmax) for limiting the x-axis range (epochs).
            ylim (tuple): Tuple (ymin, ymax) for limiting the y-axis range (loss values).
            legend_fontsize (int, optional): Font size for the legend.
        """
        # Load log data into a DataFrame
        data = self._load_log()

        plt.figure(figsize=(10, 6))
        train_data = data[data['type'] == 'train']
        val_data = data[data['type'] == 'validation']

        # Create a float epoch for smoother plotting (using global_step to approximate within an epoch)
        train_data['float_epoch'] = train_data['epoch'] + train_data['global_step'] / (train_data['global_step'].max() + 1)
        val_data['float_epoch'] = val_data['epoch'] + val_data['global_step'] / (val_data['global_step'].max() + 1)

        plt.plot(train_data['float_epoch'], train_data['loss_value'], label='Train Loss', color='blue')
        plt.plot(val_data['float_epoch'], val_data['loss_value'], label='Validation Loss', color='orange')
        plt.title('All Losses (Train and Validation)')
        plt.xlabel('Epoch (float)')
        plt.ylabel('Loss Value')
        plt.legend(fontsize=legend_fontsize)  # Set legend fontsize
        plt.grid(True)

        # Set x and y axis limits if provided
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        
        # Save the plot
        save_dir = os.path.dirname(self.log_file_path) + f"/{save_dir_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f"{save_dir}/all_losses_plot.png")
        plt.close()


# Example usage:
if __name__ == "__main__":
    logger = TSVLogger("training_log.tsv")
    
    # Logging training data with current timestamp and data size
    logger.log("train", epoch=1, global_step=100, loss_type="cross_entropy", target_name="classification", loss_value=0.456, data_size=64, accuracy=0.85, learning_rate=0.001)
    
    # Logging validation data with manually specified timestamp and data size
    logger.log("validation", epoch=1, global_step=150, loss_type="mse", target_name="regression", loss_value=0.382, data_size=128, accuracy=0.88, timestamp="2024-09-13 10:00:00")
