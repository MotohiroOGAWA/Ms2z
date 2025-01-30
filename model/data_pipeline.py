import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from typing import Dict


class BsDataset(TensorDataset):
    def __init__(self, **tensors):
        assert all(tensor.size(0) == next(iter(tensors.values())).size(0) for tensor in tensors.values()), \
            "All tensors must have the same size in the first dimension"
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)

def get_ds(
        variables: Dict[str, torch.Tensor], mode: str, 
        train_size: float = 0.8, val_size: float = 0.1, test_size: float = None, 
        batch_size: int = 8, device: torch.device = torch.device('cpu')):
    """
    Function to create datasets and dataloaders for train, validation, and test modes.
    
    Args:
        variables (Dict[str, torch.Tensor]): Input dataset variables.
        mode (str): Mode of operation ('train' or 'test').
        train_size (float): Proportion of dataset for training.
        val_size (float): Proportion of dataset for validation.
        test_size (float): Proportion of dataset for testing.
        batch_size (int): Size of batches for DataLoader.

    Returns:
        tuple: Returns dataset and corresponding DataLoaders based on mode.
    """
    # Initialize dataset using the provided variables
    variables = {key: tensor.to(device) for key, tensor in variables.items()}
    dataset = BsDataset(**variables)

    if mode == 'train':
        # Calculate sizes for train, validation, and test datasets
        train_ds_size = int(train_size * len(dataset))
        if test_size is None:
            val_ds_size = len(dataset) - train_ds_size
            test_ds_size = 0
        else:
            val_ds_size = int(val_size * len(dataset))
            test_ds_size = len(dataset) - train_ds_size - val_ds_size

        # Split the dataset into train, validation, and test datasets
        train_ds, val_ds, test_ds = random_split(dataset, [train_ds_size, val_ds_size, test_ds_size])

        # Create DataLoaders for each dataset
        train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return dataset, train_dataloader, val_dataloader, test_dataloader
    
    elif mode == 'test':
        # Only create a DataLoader for the full dataset
        test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataset, test_dataloader