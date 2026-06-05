from __future__ import annotations

from typing import Iterator, Tuple, Union, Optional, Literal, Iterable

import numpy as np

try:  # optional
    from sklearn.model_selection import StratifiedKFold
except Exception:  # pragma: no cover
    StratifiedKFold = None  # type: ignore

from .chbmit_npz import CHBMITDataset, SubsetWithInfo


def leave_one_out(
    dataset: Union[CHBMITDataset, SubsetWithInfo],
    *,
    shuffle_interictal: bool = False,
    random_state: int = 0,
) -> Iterator[Tuple[SubsetWithInfo, SubsetWithInfo]]:
    """Leave-one-positive-event-out splitter.

    This mirrors your original `leave_one_out` behavior:
    - keep only samples allowed for training (`dataset.is_used_in_train`)
    - split by `group_ids` among positive events
    - partition negative indices into the same number of folds

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

    if shuffle_interictal:
        rng = np.random.default_rng(seed=random_state)
        rng.shuffle(neg_indices)

    n_splits = len(pos_groups)
    if n_splits == 0:
        raise ValueError("No positive groups found for leave_one_out splitting")

    chunks = np.array_split(neg_indices, n_splits)
    neg_chunks = {g: np.asarray(chunks[i], dtype=int) for i, g in enumerate(pos_groups)}

    for test_group in neg_chunks.keys():
        pos_test_mask = group_id == test_group
        pos_train_mask = pos_mask & ~pos_test_mask

        pos_train_idx = np.where(pos_train_mask)[0]
        pos_test_idx = np.where(pos_test_mask)[0]

        neg_test_idx = neg_chunks[test_group]
        others = [neg_chunks[g] for g in pos_groups if g != test_group]
        neg_train_idx = np.hstack(others).astype(int) if len(others) else np.asarray([], dtype=int)

        train_idx = np.concatenate([pos_train_idx, neg_train_idx]).astype(int)
        test_idx = np.concatenate([pos_test_idx, neg_test_idx]).astype(int)

        yield SubsetWithInfo(dataset, train_idx), SubsetWithInfo(dataset, test_idx)


def leave_one_preictal(
    dataset: Union[CHBMITDataset, SubsetWithInfo],
    *,
    method: Literal["balanced"] = "balanced",
    shuffle_interictal: bool = False,
    random_state: int = 0,
) -> Iterator[Tuple[SubsetWithInfo, SubsetWithInfo]]:
    """Alias for your original "leave_one_preictal" outer-CV mode.

    Currently only method="balanced" is implemented (interictal windows are partitioned evenly).
    """

    if method != "balanced":
        raise ValueError(f"Unsupported method={method!r}. Only 'balanced' is supported.")
    yield from leave_one_out(dataset, shuffle_interictal=shuffle_interictal, random_state=random_state)


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


CVMode = Literal["leave_one_preictal", "leave_one_out", "stratified"]


def make_cv_splitter(
    dataset: Union[CHBMITDataset, SubsetWithInfo],
    *,
    mode: CVMode,
    method: Optional[str] = None,
    n_fold: int = 5,
    shuffle: bool = False,
    random_state: int = 0,
) -> Iterator[Tuple[SubsetWithInfo, SubsetWithInfo]]:
    """Compatibility helper (mirrors the old repo's make_cv_splitter API).

    Parameters
    ----------
    mode:
      - "leave_one_preictal" / "leave_one_out": outer CV split by preictal event groups
      - "stratified": inner CV split within a training subset

    method:
      - outer: "balanced" (supported)
      - inner: currently ignored (stratification already balances label ratios)
    """

    if mode in {"leave_one_preictal", "leave_one_out"}:
        yield from leave_one_preictal(
            dataset,
            method=(method or "balanced"),
            shuffle_interictal=bool(shuffle),
            random_state=int(random_state),
        )
        return

    if mode == "stratified":
        yield from stratified_kfold(
            dataset,
            n_folds=int(n_fold),
            shuffle=bool(shuffle),
            random_state=int(random_state),
        )
        return

    raise ValueError(f"Unknown CV mode: {mode}")
