import torch
from torch.utils.data import Dataset
import numpy as np

CLASSIFICATION = "classification"
REGRESSION = "regression"
TASKS = (CLASSIFICATION, REGRESSION)


def get_task(configs) -> str:
    """Task of the run: configs["data"]["task"], or a flat configs["task"] (train configs).

    Defaults to classification so configs written before regression support keep working.
    """
    configs = configs or {}
    task = (configs.get("data") or {}).get("task") or configs.get("task") or CLASSIFICATION
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Expected one of {TASKS}.")
    return task


def is_regression(configs) -> bool:
    return get_task(configs) == REGRESSION


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, task: str = CLASSIFICATION):
        """
        Args:
            X (np.ndarray): The input features.
            y (np.ndarray): The targets: class labels, or real values when task="regression".
            task (str): "classification" (targets as int64) or "regression" (targets as float32).
        """
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}'. Expected one of {TASKS}.")
        self.data = X
        self.targets = y
        self.task = task

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.targets)

    def __getitem__(self, idx: int):
        """
        Args:
            idx (int or torch.Tensor): The index or tensor index of the sample to retrieve.
        Returns:
            A tuple (data, target), where both are converted to PyTorch tensors.
        """
        # Handle torch.Tensor index for compatibility with random_split
        if isinstance(idx, torch.Tensor):
            idx = idx.item()  # Get the integer value of a single-element tensor

        # Datasets pickled before regression support have no `task` attribute.
        target_dtype = torch.float32 if getattr(self, "task", CLASSIFICATION) == REGRESSION else torch.long

        # Convert the data and target to tensors for PyTorch usage
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=target_dtype)

        return data, target
