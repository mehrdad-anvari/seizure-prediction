"""Base Dataset class"""

from __future__ import annotations
from typing import Callable, List, Optional, TypeVar
import numpy as np
import torch
from torch.utils.data import Dataset


T = TypeVar("T")  # Type variable for generic support if needed


class BaseDataset(Dataset):
    """Abstract base dataset for seizure-related data.
    
    Provides a common interface and attributes for all datasets in this module.
    Can be used for type hinting and as a mixin class.
    """

    def __init__(
        self,
        online_transforms: Optional[List[Callable]] = None,
        offline_transforms: Optional[List[Callable]] = None,
    ):
        """Initialize the base dataset with transform lists.
        
        Args:
            online_transforms: Transforms applied at runtime per sample (e.g., augmentation)
            offline_transforms: Transforms applied once to entire dataset during init
        """
        self.online_transforms = online_transforms or []
        self.offline_transforms = offline_transforms or []

    @property
    def X(self) -> np.ndarray:
        """Features array. Shape: (n_samples, n_channels, n_timesteps)"""
        raise NotImplementedError("Subclasses must implement 'X' property")

    @property
    def y(self) -> np.ndarray:
        """Labels array. Binary or multi-class labels."""
        raise NotImplementedError("Subclasses must implement 'y' property")

    @property
    def group_ids(self) -> np.ndarray:
        """Group identifiers (e.g., event IDs). Shape: (n_samples,)"""
        raise NotImplementedError("Subclasses must implement 'group_ids' property")

    @property
    def metadata(self) -> List[dict]:
        """Metadata for each sample as list of dictionaries."""
        raise NotImplementedError("Subclasses must implement 'metadata' property")


    def __len__(self) -> int:
        """Return total number of samples in the dataset."""
        raise NotImplementedError("Subclasses must implement '__len__' method")

    def __getitem__(self, idx: int):
        """Retrieve a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (features, label, metadata) where:
                - features: torch tensor of shape (n_channels, n_timesteps) or similar
                - label: torch tensor with binary/multi-class labels
                - metadata: dictionary containing sample-specific information
        """
        x = self.X[idx]
        y = self.y[idx]
        meta = self.metadata[idx]

        # Apply offline transforms once to all samples (if not already done)
        for transform in self.offline_transforms:
            if hasattr(self, '_offline_transformed'):
                break  # Already transformed
            transformed_list = []
            for i in range(len(self.X)):
                try:
                    t_x = transform(x=self.X[i])
                except TypeError:
                    raise RuntimeError(f"Transform {transform} failed on sample {i}")
                transformed_list.append(t_x)
            self.X = np.stack(transformed_list, axis=0)
            self._offline_transformed = True

        x = self.X[idx]

        # Apply online transforms per sample (e.g., augmentation)
        for transform in self.online_transforms:
            try:
                x = transform(x)
            except TypeError:
                raise RuntimeError(f"Transform {transform} failed on sample {idx}")

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            meta
        )

    def get_class_indices(self):
        """Return indices for each class.
        
        Returns:
            Tuple of (target_indices, baseline_indices) where:
                - target_indices: indices where y == 1 (or target class label)
                - baseline_indices: indices where y == 0 (background/baseline class)
            
        Subclasses can override this to handle multi-class or custom labeling.
        """
        # Default assumes binary classification with labels as 0/1 integers
        if len(self.y.shape) > 1 and self.y.shape[-1] > 1:
            # Handle one-hot encoded labels if present
            raise NotImplementedError("Subclasses must implement 'get_class_indices' for multi-class")

        target_indices = np.where(self.y == 1)[0]
        baseline_indices = np.where(self.y == 0)[0]
        
        return target_indices, baseline_indices


# Optional: Type alias for using BaseDataset in type hints
BaseDatasetType = TypeVar("BaseDatasetType", bound=BaseDataset)