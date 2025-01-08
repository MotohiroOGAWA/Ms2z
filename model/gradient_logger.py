import torch
import matplotlib.pyplot as plt
import dill

class GradientLogger:
    """
    GradientLogger class to log and visualize gradients of specified parameters in a PyTorch model.
    """
    def __init__(self, file, save_interval=100):
        """
        Initialize the GradientLogger.
        """
        self.gradients = {}  # Dictionary to store gradients for specified parameters
        self.steps = []      # List to store global steps
        self.epochs = []     # List to store epochs corresponding to steps
        self.save_interval = save_interval
        self.file = file

    def __len__(self):
        return len(self.steps)
        

    def log(self, model, step, epoch):
        """
        Log gradients of specified parameters in the model.

        Args:
            model (torch.nn.Module): The PyTorch model.
        """
        self._log_step(step, epoch)
        for name, param in model.named_parameters():
            if name not in self.gradients:
                self.gradients[name] = [0.0] * (len(self.steps)-1)
            if param.grad is not None:
                self.gradients[name].append(param.grad.norm().item())
            else:
                self.gradients[name].append(0.0)
        
        if len(self) % self.save_interval == 0:
            self.save(self.file)

    def _log_step(self, step, epoch):
        """
        Log the current global step and corresponding epoch.

        Args:
            step (int): The global step value.
            epoch (float): The epoch value (can be fractional).
        """
        self.steps.append(step)
        self.epochs.append(epoch)


    def plot_gradients(self):
        """
        Plot gradients for each parameter individually over global steps.
        """
        if not self.gradients:
            print("No gradients recorded. Please ensure gradients are being tracked.")
            return

        for name, values in self.gradients.items():
            plt.figure(figsize=(10, 6))

            # Plot gradients for the current parameter
            plt.plot(self.steps, values, label=f"Gradient of {name}")
            
            # Configure the plot
            plt.xlabel("Global Step")
            plt.ylabel("Gradient Norm")
            plt.title(f"Gradient Norm vs. Global Steps for {name}")
            plt.legend()
            plt.grid(True)
            
            # Show the plot
            plt.show()

    def save(self, file_path):
        """
        Save the current state of the GradientLogger to a file.

        Args:
            file_path (str): Path to the file where the logger will be saved.
        """
        with open(file_path, "wb") as file:
            dill.dump(self, file)

    @staticmethod
    def load(file_path):
        """
        Load a GradientLogger from a file.

        Args:
            file_path (str): Path to the file where the logger is stored.

        Returns:
            GradientLogger: The loaded GradientLogger instance.
        """
        with open(file_path, "rb") as file:
            return dill.load(file)