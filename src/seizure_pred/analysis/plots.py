from __future__ import annotations

import json
from typing import Mapping, Optional

import numpy as np


def _mpl():
    # Lazy import: allows analysis to run without matplotlib installed unless plotting is used.
    # Force a non-interactive backend for headless/CI environments.
    import matplotlib  # type: ignore
    try:
        matplotlib.use("Agg", force=True)  # type: ignore
    except Exception:
        pass
    import matplotlib.pyplot as plt  # type: ignore
    return plt


def plot_history(history_jsonl: str, *, save_path: str) -> None:
    plt = _mpl()

    epochs = []
    train_loss = []
    val_loss = []

    with open(history_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            epochs.append(int(row.get("epoch", len(epochs) + 1)))
            if "train_loss" in row:
                train_loss.append(float(row["train_loss"]))
            if "val_loss" in row:
                val_loss.append(float(row["val_loss"]))

    plt.figure()
    if train_loss:
        plt.plot(epochs[: len(train_loss)], train_loss, label="train_loss")
    if val_loss:
        plt.plot(epochs[: len(val_loss)], val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion(confusion, *, save_path: str) -> None:
    plt = _mpl()
    c = np.asarray(confusion, dtype=float)

    plt.figure()
    plt.imshow(c, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    for (i, j), v in np.ndenumerate(c):
        plt.text(j, i, f"{int(v)}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, *, save_path: str, auc: Optional[float] = None) -> None:
    plt = _mpl()
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}" if auc is not None else "ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", label="chance")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_pr(rec: np.ndarray, prec: np.ndarray, *, save_path: str, auc: Optional[float] = None) -> None:
    plt = _mpl()
    plt.figure()
    plt.plot(rec, prec, label=f"AUC={auc:.4f}" if auc is not None else "PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_preictal_prob(
    x: np.ndarray,
    ensemble_prob: np.ndarray,
    inner_prob: Mapping[str, np.ndarray],
    *,
    save_path: str,
    title: Optional[str] = None,
    event_type: str = "preictal",
) -> None:
    """Compare an outer-fold ensemble with its inner-fold probabilities."""
    plt = _mpl()
    x = np.asarray(x, dtype=np.float64)
    ensemble_prob = np.asarray(ensemble_prob, dtype=np.float64)

    if x.ndim != 1 or ensemble_prob.ndim != 1:
        raise ValueError("x and ensemble_prob must be one-dimensional")
    if x.size != ensemble_prob.size:
        raise ValueError("x and ensemble_prob must have the same length")

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, values in inner_prob.items():
        prob = np.asarray(values, dtype=np.float64)
        if prob.ndim != 1 or prob.size != x.size:
            plt.close(fig)
            raise ValueError(f"{name} probabilities must have the same length as x")
        ax.plot(x, prob, linewidth=1.0, alpha=0.65, label=name)

    ax.plot(
        x,
        ensemble_prob,
        color="black",
        linewidth=2.5,
        label="outer ensemble",
        zorder=10,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(f"Time within {event_type} event (minutes)")
    ax.set_ylabel("Predicted preictal probability")
    ax.set_title(title or "Preictal probability comparison")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_interictal_prob(
    x: np.ndarray,
    ensemble_prob: np.ndarray,
    inner_prob: Mapping[str, np.ndarray],
    *,
    save_path: str,
    title: Optional[str] = None,
) -> None:
    """Compare predictions on interictal windows across nested-CV folds."""
    plot_preictal_prob(
        x,
        ensemble_prob,
        inner_prob,
        save_path=save_path,
        title=title,
        event_type="interictal",
    )


def plot_interictal_combined(
    events: list,
    *,
    save_path: str,
    title: Optional[str] = None,
) -> None:
    """Plot all interictal events in a single figure with red boundary lines.

    Each element in ``events`` is a dict with keys:

    - ``x_index``: ``np.ndarray`` of cumulative sample indices
    - ``ensemble_prob``: ``np.ndarray``
    - ``inner_prob``: ``Mapping[str, np.ndarray]``
    - ``label``: ``str`` event label (used only for debugging)

    A red dashed vertical line is drawn at every event boundary.
    """
    plt = _mpl()

    if not events:
        raise ValueError("events must be non-empty")

    fig, ax = plt.subplots(figsize=(12, 5))

    # Collect all inner-split names so each gets one consistent colour.
    all_inner_names: list[str] = []
    for ev in events:
        for name in ev["inner_prob"]:
            if name not in all_inner_names:
                all_inner_names.append(name)

    # Plot each event's data.
    event_boundaries: list[float] = []
    for i, ev in enumerate(events):
        x = np.asarray(ev["x_index"], dtype=np.float64)
        ensemble_prob = np.asarray(ev["ensemble_prob"], dtype=np.float64)

        if x.ndim != 1 or ensemble_prob.ndim != 1:
            plt.close(fig)
            raise ValueError("x_index and ensemble_prob must be one-dimensional")
        if x.size != ensemble_prob.size:
            plt.close(fig)
            raise ValueError(
                f"x_index and ensemble_prob must have the same length "
                f"(event {i}: {x.size} vs {ensemble_prob.size})"
            )

        # Inner-split dots (scatter to reduce noise).
        for name in all_inner_names:
            values = ev["inner_prob"].get(name)
            if values is None:
                continue
            prob = np.asarray(values, dtype=np.float64)
            if prob.ndim != 1 or prob.size != x.size:
                plt.close(fig)
                raise ValueError(
                    f"{name} probabilities must have the same length as x "
                    f"(event {i})"
                )
            ax.scatter(x, prob, s=4, alpha=0.4, label=name)

        # Ensemble dots (black, more visible).
        ax.scatter(
            x, ensemble_prob,
            color="black", s=6, alpha=0.7, label="outer ensemble", zorder=10,
        )

        # Mark boundary after every event except the last.
        if i < len(events) - 1:
            boundary = float(x[-1])
            event_boundaries.append(boundary)

    # Draw thick red boundary markers at the top of the plot.
    for boundary in event_boundaries:
        ax.axvline(
            x=boundary, color="red", linestyle="-", linewidth=5.0, alpha=1.0)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Sample index (cumulative)")
    ax.set_ylabel("Predicted preictal probability")
    ax.set_title(title or "Interictal probability — all events combined")

    # Reduce tick density on x-axis so labels don't overlap.
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=12))

    # Deduplicate legend entries (each inner-split name appears once).
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    unique = []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen.add(lab)
            unique.append((h, lab))
    if unique:
        ax.legend(
            [h for h, _ in unique],
            [lab for _, lab in unique],
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_prob_vs_pp_scatter_combined(
    interictal: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    preictal: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    *,
    save_path: str,
    title: Optional[str] = None,
) -> None:
    """Scatter plot with interictal and preictal on the same axes.

    Each tuple is ``(prob, pp_max, pp_mean)`` — model probability and EEG
    peak-to-peak features from the meta dict of predictions.jsonl.

    Points are colour-coded by event type (interictal=blue, preictal=red).

    Two subplots side by side:
    - Left:  prob vs pp_max
    - Right: prob vs pp_mean
    """
    plt = _mpl()

    fig, (ax_max, ax_mean) = plt.subplots(1, 2, figsize=(14, 6))

    scatter_kw: dict = dict(s=14, alpha=0.55, edgecolors="none")

    # ---- pp_max subplot ----
    if interictal is not None:
        ax_max.scatter(
            interictal[0], interictal[1],
            color="steelblue", label="interictal", **scatter_kw,
        )
    if preictal is not None:
        ax_max.scatter(
            preictal[0], preictal[1],
            color="firebrick", label="preictal", **scatter_kw,
        )
    ax_max.set_xlim(0.0, 1.0)
    ax_max.set_xlabel("Model probability")
    ax_max.set_ylabel("pp_max (EEG peak-to-peak max)")
    ax_max.set_title("Prob vs pp_max")
    ax_max.legend(loc="lower right")

    # ---- pp_mean subplot ----
    if interictal is not None:
        ax_mean.scatter(
            interictal[0], interictal[2],
            color="steelblue", label="interictal", **scatter_kw,
        )
    if preictal is not None:
        ax_mean.scatter(
            preictal[0], preictal[2],
            color="firebrick", label="preictal", **scatter_kw,
        )
    ax_mean.set_xlim(0.0, 1.0)
    ax_mean.set_xlabel("Model probability")
    ax_mean.set_ylabel("pp_mean (EEG peak-to-peak mean)")
    ax_mean.set_title("Prob vs pp_mean")
    ax_mean.legend(loc="lower right")

    fig.suptitle(title or "Prob vs EEG P-P features — interictal + preictal")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# Backward-compatible name for callers using the original longer helper name.
plot_preictal_probability_comparison = plot_preictal_prob
