"""CHB-MIT NPZ dataset loader (ported from original repository).

This module is used at training/inference time and therefore avoids heavy preprocessing
dependencies (MNE). Preprocessing utilities live in seizure_pred.preprocessing.
"""

import torch
import numpy as np
import pandas as pd
from typing import Callable, List, Literal
import glob
import os
from seizure_pred.data.uint16 import invert_uint16_scaling
from tqdm import tqdm
from collections import Counter
from seizure_pred.data.base_dataset import BaseDataset

class CHBMITDataset(BaseDataset):
    def __init__(
        self,
        dataset_dir: str = "data/BIDS_CHB-MIT",
        use_uint16: bool = False,
        subject_id: str = "01",
        online_transforms: List[Callable] = None,
        offline_transforms: List[Callable] = None,
        suffix: str = "fd_5s_szx5_prex5",
        task: Literal["prediction", "detection"] = "prediction",
        print_events: bool = True,
    ):
        """
        This class does three things:
        1. Loads the processed BIDS_CHB-MIT data and its metadata from .npz files for the specified subject
        2. Determine the binary labels and which samples are used for training
        3. Applies offline and online transforms to the data

        metadata fields:
        "event_id", "label", "epoch_index_within_event", "global_epoch_id"
        "n_segments_in_event", "start_time_in_event", "augmented", "pp_mean", "pp_max"
        "sd_mean", "sd_max", "onset_sec", "duration_sec"

        Args:
            dataset_dir (string): path to the processed BIDS_CHB-MIT dataset
            use_uint16 (boolean): if true uses uint16 format
            subject_id (str): use "*" to include all subjects,
        """
        # Task settings
        if task == "detection":
            target_label = "seizure"
            background_labels_training = [
                "interictal",
                "preictal",
                "post_buffer",
                "pre_buffer",
            ]
        else:
            target_label = "preictal"
            background_labels_training = ["interictal"]

        # Subject directory
        subject_dir = os.path.join(dataset_dir, f"sub-{subject_id}/")
        print(f"Loading data for subject: {subject_id} from {subject_dir}")
        if not os.path.isdir(subject_dir):
            raise ValueError(f"Subject directory not found: {subject_dir}")

        # Find the NPZ session files
        suffix_pattern = (
            f"*{suffix}_uint16.npz" if use_uint16 else f"*{suffix}_float.npz"
        )
        search_pattern = os.path.join(subject_dir, "ses-*", "eeg", suffix_pattern)
        ses_paths = glob.glob(search_pattern)
        print(f"Found {len(ses_paths)} session files using pattern: {search_pattern}")

        if len(ses_paths) == 0:
            raise ValueError(
                f"No processed NPZ files found for subject_id={subject_id}"
            )

        all_X, all_y, all_group_ids, all_metadata = [], [], [], []

        # Load each session
        for i, ses_path in enumerate(ses_paths):
            data = np.load(ses_path, allow_pickle=True)

            # Load X
            X_temp = data["X"]
            if use_uint16:
                scales = data["scales"]
                X_temp = invert_uint16_scaling(X_temp, scales)
            all_X.append(X_temp)

            # Load labels
            all_y.append(data["y"])

            # Load metadata
            meta_obj = data["meta_df"]
            meta_dict = meta_obj.item() if hasattr(meta_obj, "item") else dict(meta_obj)
            meta_df = pd.DataFrame(meta_dict)

            # event_id is used as a group label
            event_ids = [f"{v}_{i}" for v in meta_df["event_id"].values]
            all_group_ids.append(event_ids)

            # store list of dicts
            all_metadata.extend(meta_df.to_dict(orient="records"))

        # Concatenate sessions
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        group_ids = np.concatenate(all_group_ids, axis=0)

        if print_events:
            counter = Counter(group_ids.tolist())
            print("\nData samples per event (group_id):")
            for g in sorted(counter.keys()):
                print(f"  {g:<12}: {counter[g]}")
            print("")

        if len(y) == 0:
            raise ValueError(
                f"No data samples found for subject_id={subject_id} in {dataset_dir}"
            )

        # Prepare tensors and labels
        self.X = X.astype(np.float32)

        # Convert textual labels to 0/1
        self.y = np.array(
            [1 if label == target_label else 0 for label in y],
            dtype=np.int64,
        )

        self.group_ids = group_ids
        self.metadata = all_metadata

        # Which samples are used in training
        allowed = {target_label} | set(background_labels_training)
        self.is_used_in_train = np.array([label in allowed for label in y])

        # Offline transforms (applied once)
        self.online_transform = online_transforms or []
        for transform in offline_transforms or []:
            transformed = []
            for i in tqdm(range(self.X.shape[0]), desc="Applying offline transforms"):
                transformed.append(transform(eeg=self.X[i]))
            self.X = np.stack(transformed, axis=0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        meta = self.metadata[idx]

        for transform in self.online_transform:
            x = transform(x)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            meta,
        )

    def get_class_indices(self):
        """Return indices for each class"""
        target_indices = np.where(self.y == 1)[0]
        baseline_indices = np.where(self.y == 0)[0]
        return target_indices, baseline_indices
