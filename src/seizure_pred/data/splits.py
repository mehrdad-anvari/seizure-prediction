from __future__ import annotations

from typing import Iterator, Tuple, Union, Optional, Literal

import numpy as np

try:  # optional
    from sklearn.model_selection import StratifiedKFold
except Exception:  # pragma: no cover
    StratifiedKFold = None  # type: ignore

from .chbmit_npz import CHBMITDataset
from .base_dataset import BaseDataset
from torch.utils.data import Subset



class SubsetWithInfo(Subset):
    """A Subset that maintains instances' classes and groups information"""

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        if isinstance(dataset, SubsetWithInfo):
            self.base_dataset: BaseDataset = dataset.base_dataset
            self.base_indices = np.array(dataset.base_indices)[indices]
        else:
            self.base_dataset: BaseDataset = dataset
            self.base_indices = indices

        self.y = self.base_dataset.y[self.base_indices]
        self.group_ids = self.base_dataset.group_ids[self.base_indices]
        self.metadata = [self.base_dataset.metadata[i] for i in self.base_indices]

    def get_class_indices(self):
        """Return indices for each class within this subset"""
        target_indices = np.where(self.y == 1)[0]
        baseline_indices = np.where(self.y == 0)[0]
        return target_indices, baseline_indices


def original_only(dataset: Union[CHBMITDataset, SubsetWithInfo]) -> SubsetWithInfo:
    """Return a subset containing only non-augmented samples.

    Validation and test metrics should represent the original fixed-length
    windows, not the extra overlapping windows created by preprocessing.
    Missing ``augmented`` metadata is treated as original for older datasets.
    """
    from seizure_pred.training.engine.metrics import is_original_segment_meta

    indices = [i for i, meta in enumerate(dataset.metadata) if is_original_segment_meta(meta)]
    return SubsetWithInfo(dataset, np.asarray(indices, dtype=int))
    
def leave_one_out(
    dataset: Union[CHBMITDataset, SubsetWithInfo],
    *,
    method: Literal["balanced", "balanced_shuffled", "nearest"] = "balanced",
    shuffle_interictal: bool = False,
    random_state: int = 0,
) -> Iterator[Tuple[SubsetWithInfo, SubsetWithInfo]]:
    """Leave-one-positive-event-out splitter.

    This mirrors the original ``leave_one_out`` / ``leave_one_preictal`` behavior:
    - keep only samples allowed for training (``dataset.is_used_in_train``)
    - split by ``group_ids`` among positive (preictal/seizure) events
    - partition negative (interictal) indices into the same number of folds

    Methods
    -------
    - ``balanced``: partition negatives evenly (chronological order) across folds.
    - ``balanced_shuffled``: same as balanced but negatives are shuffled first
      (``shuffle_interictal=True`` is equivalent).
    - ``nearest``: the test negatives for each fold are the temporally nearest
      interictal windows to the held-out positive event (reduces the
      train/test distribution mismatch for time-adjacent data).

    Yielded:
        (train_subset, test_subset)

    Notes
    -----
    Shuffling interictal is usually discouraged for moving-window evaluation.
    """
    if hasattr(dataset, "is_used_in_train"):
        mask = getattr(dataset, "is_used_in_train")
        if mask is not None:
            dataset = SubsetWithInfo(dataset, np.where(np.asarray(mask))[0])

    y = np.asarray(dataset.y)
    group_id = np.asarray(dataset.group_ids)

    pos_mask = y == 1
    neg_mask = ~pos_mask

    pos_groups = np.unique(group_id[pos_mask])
    neg_indices = np.where(neg_mask)[0]

    rng = np.random.default_rng(seed=random_state)
    if method == "balanced_shuffled":
        rng.shuffle(neg_indices)

    n_splits = len(pos_groups)
    if n_splits == 0:
        raise ValueError("No positive groups found for leave_one_out splitting")

    base_indices = getattr(dataset, "base_indices", None)
    base_indices = np.asarray(base_indices) if base_indices is not None else np.arange(len(y))

    # Build the per-fold negative test assignment.
    if method == "nearest":
        neg_base = base_indices[neg_indices]
        neg_test_for_group: dict = {}
        n_per_fold = max(1, len(neg_indices) // n_splits)
        for g in pos_groups:
            pos_test_idx = np.where((group_id == g) & pos_mask)[0]
            if len(pos_test_idx) == 0:
                neg_test_for_group[g] = np.array([], dtype=int)
                continue
            pos_base = base_indices[pos_test_idx]
            # distance from each negative to the nearest positive window
            dist = np.min(np.abs(neg_base[:, None] - pos_base[None, :]), axis=1)
            nearest_order = np.argsort(dist, kind="stable")
            neg_test_for_group[g] = neg_indices[nearest_order[:n_per_fold]]
        # train negatives = all negatives not selected as any test fold
        all_test_neg = np.unique(np.concatenate([neg_test_for_group[g] for g in pos_groups]))
        neg_train_global = np.setdiff1d(neg_indices, all_test_neg)
    else:
        chunks = np.array_split(neg_indices, n_splits)
        neg_test_for_group = {g: np.asarray(chunks[i], dtype=int) for i, g in enumerate(pos_groups)}

    for test_group in pos_groups:
        # Robust to datasets where positive/negative samples share a group_id:
        # only positives of the held-out event form the test "positive" set.
        pos_test_mask = pos_mask & (group_id == test_group)
        pos_train_mask = pos_mask & (group_id != test_group)

        pos_train_idx = np.where(pos_train_mask)[0]
        pos_test_idx = np.where(pos_test_mask)[0]

        neg_test_idx = neg_test_for_group[test_group]
        if method == "nearest":
            neg_train_idx = neg_train_global
        else:
            others = [neg_test_for_group[g] for g in pos_groups if g != test_group]
            neg_train_idx = np.hstack(others).astype(int) if len(others) else np.asarray([], dtype=int)

        train_idx = np.concatenate([pos_train_idx, neg_train_idx]).astype(int)
        test_idx = np.concatenate([pos_test_idx, neg_test_idx]).astype(int)

        yield SubsetWithInfo(dataset, train_idx), SubsetWithInfo(dataset, test_idx)


def leave_one_preictal(
    dataset: Union[CHBMITDataset, SubsetWithInfo],
    *,
    method: Literal["balanced", "balanced_shuffled", "nearest"] = "balanced",
    shuffle_interictal: bool = False,
    random_state: int = 0,
) -> Iterator[Tuple[SubsetWithInfo, SubsetWithInfo]]:
    """Alias for the original "leave_one_preictal" outer-CV mode.

    Methods
    -------
    - ``balanced``: partition interictal windows evenly across folds.
    - ``balanced_shuffled``: balanced + randomized selection of interictal windows.
    - ``nearest``: use temporally nearest interictal windows to each held-out event.
    """
    if shuffle_interictal and method == "balanced":
        method = "balanced_shuffled"
    yield from leave_one_out(
        dataset,
        method=method,
        shuffle_interictal=shuffle_interictal,
        random_state=random_state,
    )


def stratified_kfold(
    dataset: Union[CHBMITDataset, SubsetWithInfo],
    *,
    n_folds: int = 5,
    shuffle: bool = False,
    random_state: int = 0,
) -> Iterator[Tuple[SubsetWithInfo, SubsetWithInfo]]:
    """Stratified K-Fold splitter (inner-CV).

    This keeps the label distribution roughly constant across folds.

    Notes
    -----
    - Requires scikit-learn. If unavailable, raises a clear ImportError.
    - Operates on the provided dataset indices (i.e., if you pass the outer-train subset,
      it will split within that subset).
    """

    if StratifiedKFold is None:
        raise ImportError("StratifiedKFold requires scikit-learn. Install with: pip install scikit-learn")

    y = np.asarray(dataset.y)
    skf = StratifiedKFold(n_splits=int(n_folds), shuffle=bool(shuffle), random_state=int(random_state) if shuffle else None)
    idx_all = np.arange(len(y))
    for train_idx, val_idx in skf.split(idx_all, y):
        yield SubsetWithInfo(dataset, train_idx.astype(int)), SubsetWithInfo(dataset, val_idx.astype(int))


def split_into_strata(indices, N=5, M=10):
    """Split indices into N strata, with M samples per stratum in each iteration."""
    import math
    max_M = math.ceil(len(indices) / (N + 1))
    if M > max_M:
        raise ValueError(
            f"M={M} is too large for event length={len(indices)} and N={N} folds. "
            f"Maximum valid M is {max_M}."
        )

    indices = np.sort(indices)
    if len(indices) == 0:
        return [np.array([]) for _ in range(N)]

    splits = [[] for _ in range(N)]
    i = 0

    while i < len(indices):
        for fold in range(N):
            if i >= len(indices):
                break
            end = min(i + M, len(indices))
            splits[fold].extend(indices[i:end])
            i = end

    return [np.array(s) for s in splits if len(s) > 0]


def KFold(
    dataset,
    shuffle=True,
    n_fold=5,
    random_state=0,
    mode: str = "per_event_strata",
    M=10,
):
    """Custom K-Fold splitter supporting stratified chronological splits."""
    rng = np.random.RandomState(random_state)

    y = np.array(dataset.y)
    group_ids = np.array(dataset.group_ids)
    base_indices = np.arange(len(dataset))

    # Separate classes
    class0 = base_indices[y == 0]
    class1 = base_indices[y == 1]

    # Unique seizure groups
    seizure_groups = np.unique(group_ids[y == 1])
    background_groups = np.unique(group_ids[y == 0])

    # ---------- MODE 1: RANDOM ----------
    if mode == "random_split" or mode == "split":
        if mode == "random_split":
            rng.shuffle(class0)
            rng.shuffle(class1)

        cls0_splits = np.array_split(class0, n_fold)
        cls1_splits = np.array_split(class1, n_fold)

    # ---------- MODE 2: STRATA ----------
    elif mode == "strata":
        cls0_splits = split_into_strata(class0, N=n_fold, M=M)
        cls1_splits = split_into_strata(class1, N=n_fold, M=M)

    # ---------- MODE 3: PER-EVENT STRATA ----------
    elif mode == "per_event_strata":
        # Split background events one-by-one
        bg_splits = []
        for gid in background_groups:
            inds = base_indices[(group_ids == gid) & (y == 0)]
            bg_splits.append(split_into_strata(inds, N=n_fold, M=M))

        # Combine background stratification across events
        combined_bg_splits = []
        for k in range(n_fold):
            combined_bg_splits.append(np.concatenate([event[k] for event in bg_splits if len(event) > k]))

        # Split seizure events one-by-one
        sz_splits = []
        for gid in seizure_groups:
            inds = base_indices[(group_ids == gid) & (y == 1)]
            sz_splits.append(split_into_strata(inds, N=n_fold, M=M))

        # Combine seizure stratification across events
        combined_sz_splits = []
        for k in range(n_fold):
            combined_sz_splits.append(np.concatenate([event[k] for event in sz_splits if len(event) > k]))

        cls0_splits = combined_bg_splits
        cls1_splits = combined_sz_splits

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Use random_split, split, strata, per_event_strata."
        )

    for k in range(n_fold):
        # Guard against index errors
        cls0_val = cls0_splits[k] if k < len(cls0_splits) else np.array([], dtype=int)
        cls1_val = cls1_splits[k] if k < len(cls1_splits) else np.array([], dtype=int)
        
        val_idx = np.concatenate([cls0_val, cls1_val]).astype(int)
        
        cls0_train_parts = [cls0_splits[i] for i in range(len(cls0_splits)) if i != k]
        cls1_train_parts = [cls1_splits[i] for i in range(len(cls1_splits)) if i != k]
        
        cls0_train = np.concatenate(cls0_train_parts) if cls0_train_parts else np.array([], dtype=int)
        cls1_train = np.concatenate(cls1_train_parts) if cls1_train_parts else np.array([], dtype=int)
        
        train_idx = np.concatenate([cls0_train, cls1_train]).astype(int)

        if shuffle:
            train_idx = rng.permutation(train_idx)
            val_idx = rng.permutation(val_idx)

        yield (SubsetWithInfo(dataset, train_idx), SubsetWithInfo(dataset, val_idx))


def make_cv_splitter(
    dataset: Union[BaseDataset, SubsetWithInfo],
    *,
    mode: Optional[str] = None,
    method: Optional[str] = None,
    n_folds: Optional[int] = None,
    n_fold: Optional[int] = None,
    shuffle: bool = False,
    random_state: int = 0,
    M: int = 10,
) -> Iterator[Tuple[SubsetWithInfo, SubsetWithInfo]]:
    """Helper to perform dataset splits for cross-validation."""
    # Filter dataset first to keep only samples allowed for training
    if hasattr(dataset, "is_used_in_train"):
        mask = getattr(dataset, "is_used_in_train")
        if mask is not None:
            dataset = SubsetWithInfo(dataset, np.where(np.asarray(mask))[0])

    resolved_method = method
    if resolved_method is None:
        if mode in {"leave_one_preictal", "leave_one_out", "LOO"}:
            resolved_method = "LOO"
        elif mode == "stratified":
            resolved_method = "stratified"
        elif mode == "KFold":
            resolved_method = "KFold"
        else:
            resolved_method = "LOO"  # fallback

    resolved_n_folds = n_folds if n_folds is not None else (n_fold if n_fold is not None else 5)

    if resolved_method == "LOO":
        loo_method = "balanced"
        if mode in {"balanced", "balanced_shuffled", "nearest"}:
            loo_method = mode
        elif shuffle:
            loo_method = "balanced_shuffled"
        yield from leave_one_out(
            dataset,
            method=loo_method,
            shuffle_interictal=shuffle,
            random_state=random_state,
        )
    elif resolved_method == "stratified":
        yield from stratified_kfold(
            dataset,
            n_folds=resolved_n_folds,
            shuffle=shuffle,
            random_state=random_state
        )
    elif resolved_method == "KFold":
        kfold_mode = mode if mode in {"random_split", "split", "strata", "per_event_strata"} else "per_event_strata"
        yield from KFold(
            dataset,
            shuffle=shuffle,
            n_fold=resolved_n_folds,
            random_state=random_state,
            mode=kfold_mode,
            M=M,
        )
    else:
        raise ValueError(f"Unknown CV method: {resolved_method}")
