from __future__ import annotations

import argparse
import json
import os
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from seizure_pred.core.config import TrainConfig
from seizure_pred.core.io import load_dict, merge_dict, from_dict
from seizure_pred.core.seed import seed_everything
from seizure_pred.core.validate import validate_config_dict
from seizure_pred.training.registries import CALLBACKS, DATALOADERS, MODELS, LOSSES, OPTIMIZERS, SCHEDULERS
from seizure_pred.training.engine.pipeline import build_dataset, iter_splits, build_loader
from seizure_pred.training.engine.artifacts import ArtifactWriter
from seizure_pred.training.engine.trainer import Trainer
from seizure_pred.training.engine.trainer_mil import TrainerMIL
from seizure_pred.core.logging import setup_logging


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _make_run_dir(cfg: TrainConfig, stamp: str, split_index: int) -> str:
    # <save_dir>/<run_name>/<stamp>/split_<k>/
    root = os.path.join(cfg.save_dir, cfg.run_name, stamp, f"split_{split_index}")
    os.makedirs(root, exist_ok=True)
    return root


def add_train_cmd(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="Train a model (prediction/detection/MIL)")
    p.add_argument("--config", required=True, help="YAML/JSON config file")
    p.add_argument("--override", default=None, help="Optional YAML/JSON override file merged on top")
    p.add_argument("--n-folds", type=int, default=5, help="Number of folds for leave-one-out style split")
    p.add_argument("--dataloader", default=None, help="Override dataloader strategy name")
    p.add_argument("--mil", action="store_true", help="Use MIL trainer")
    p.add_argument("--strict", action="store_true", help="Fail fast if requested components are missing")
    p.add_argument("--print-config", action="store_true", help="Print merged config and exit")

    # Nested CV overrides
    p.add_argument("--outer-method", default=None, help="Outer CV method")
    p.add_argument("--inner-method", default=None, help="Inner CV method")
    p.add_argument("--outer-n-fold", type=int, default=None, help="Outer number of folds")
    p.add_argument("--inner-n-fold", type=int, default=None, help="Inner number of folds")
    p.add_argument("--outer-shuffle", dest="outer_shuffle", action="store_true", help="Shuffle outer folds")
    p.add_argument("--no-outer-shuffle", dest="outer_shuffle", action="store_false", help="Don't shuffle outer folds")
    p.set_defaults(outer_shuffle=None)
    p.add_argument("--inner-shuffle", dest="inner_shuffle", action="store_true", help="Shuffle inner folds")
    p.add_argument("--no-inner-shuffle", dest="inner_shuffle", action="store_false", help="Don't shuffle inner folds")
    p.set_defaults(inner_shuffle=None)

    p.set_defaults(func=run_train)


def run_train(args: argparse.Namespace) -> None:
    raw = load_dict(args.config)
    if args.override:
        raw = merge_dict(raw, load_dict(args.override))

    # Validate user config dict against TrainConfig schema (clear errors)
    validate_config_dict(raw, TrainConfig)

    cfg: TrainConfig = from_dict(TrainConfig, raw)

    # Register built-in plugins only when training is invoked (avoids heavy imports on `--help`).
    import seizure_pred.training as training
    training.register_all()
    import seizure_pred.models as models
    models.register_all()
    
    # Optional overrides
    if args.dataloader is not None:
        cfg.data.kwargs = dict(cfg.data.kwargs or {})
        cfg.data.kwargs["dataloader_name"] = args.dataloader  # keep provenance
    dl_name = cfg.data.dataloader_type or "torch"

    if args.print_config:
        print(json.dumps(asdict(cfg), indent=2, default=str))
        return

    # Resolve CV command-line overrides
    from seizure_pred.core.config import CvConfig
    any_cv_arg = (
        args.outer_method is not None
        or args.inner_method is not None
        or args.outer_n_fold is not None
        or args.inner_n_fold is not None
        or args.outer_shuffle is not None
        or args.inner_shuffle is not None
    )
    if any_cv_arg and cfg.cv is None:
        cfg.cv = CvConfig()

    if cfg.cv is not None:
        if args.outer_method is not None:
            cfg.cv.outer_method = args.outer_method
        if args.inner_method is not None:
            cfg.cv.inner_method = args.inner_method
        if args.outer_n_fold is not None:
            cfg.cv.outer_n_fold = args.outer_n_fold
        if args.inner_n_fold is not None:
            cfg.cv.inner_n_fold = args.inner_n_fold
        if args.outer_shuffle is not None:
            cfg.cv.outer_shuffle = args.outer_shuffle
        if args.inner_shuffle is not None:
            cfg.cv.inner_shuffle = args.inner_shuffle

    # Build dataset + split
    dataset = build_dataset(cfg)
    stamp = _utc_stamp()

    if cfg.cv is not None:
        run_nested_cv(cfg, dataset, stamp, args)
        return

    splits = list(iter_splits(dataset, cfg.data))
    if not splits:
        raise SystemExit("No splits found to train on.")

    try:
        for split_index, (train_set, val_set) in enumerate(splits):
            # Determinism / seeding
            seed_everything(getattr(cfg, "determinism", None), seed=cfg.seed)

            # Run directory + logger
            run_dir = _make_run_dir(cfg, stamp, split_index)
            logger = setup_logging(run_dir)
            logger.info("[train] Starting split %d/%d", split_index + 1, len(splits))
            logger.info("[train] run_dir=%s", run_dir)
            logger.info("[train] config_path=%s", args.config)
            if args.override:
                logger.info("[train] override_path=%s", args.override)

            # Build loaders via registry (factory pattern)
            # (If strict and not present -> raise with helpful registry error)
            if dl_name not in DATALOADERS:
                raise SystemExit(f"Unknown dataloader '{dl_name}'. Use `seizure-pred list`.")

            train_loader = build_loader(dl_name, train_set, cfg, shuffle=True)
            val_loader = build_loader(dl_name, val_set, cfg, shuffle=False)

            # Build model/loss/optim/sched from registries
            if args.strict and cfg.model.name not in MODELS:
                raise SystemExit(f"Unknown model '{cfg.model.name}'. Use `seizure-pred list`.")
            if args.strict and cfg.loss.name not in LOSSES:
                raise SystemExit(f"Unknown loss '{cfg.loss.name}'. Use `seizure-pred list`.")
            if args.strict and cfg.optim.name not in OPTIMIZERS:
                raise SystemExit(f"Unknown optimizer '{cfg.optim.name}'. Use `seizure-pred list`.")
            if args.strict and cfg.sched.name and cfg.sched.name not in SCHEDULERS:
                raise SystemExit(f"Unknown scheduler '{cfg.sched.name}'. Use `seizure-pred list`.")

            model = MODELS.create(cfg.model.name, cfg.model)
            loss_fn = LOSSES.create(cfg.loss.name, **(cfg.loss.kwargs or {}))
            optimizer = OPTIMIZERS.create(
                cfg.optim.name,
                model.parameters(),
                lr=cfg.optim.lr,
                weight_decay=cfg.optim.weight_decay,
                **(cfg.optim.kwargs or {}),
            )
            scheduler = None
            if cfg.sched.name:
                scheduler = SCHEDULERS.create(cfg.sched.name, optimizer, **(cfg.sched.kwargs or {}))

            # Callbacks: cfg.callbacks = [{name, kwargs}, ...]
            cb_list = []
            for item in getattr(cfg, "callbacks", []) or []:
                name = item["name"] if isinstance(item, dict) else getattr(item, "name", None)
                kwargs = item.get("kwargs", {}) if isinstance(item, dict) else dict(getattr(item, "kwargs", {}) or {})
                if not name:
                    continue
                if args.strict and name not in CALLBACKS:
                    raise SystemExit(f"Unknown callback '{name}'. Use `seizure-pred list`.")
                cb_list.append(CALLBACKS.create(name, **kwargs))

            # Artifact writer
            writer = ArtifactWriter(run_dir)
            writer.write_schema()
            writer.write_config(asdict(cfg))

            # Train
            if args.mil:
                trainer = TrainerMIL(
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cfg=cfg,
                    run_dir=run_dir,
                    artifact_writer=writer,
                    callbacks=cb_list,
                )
            else:
                trainer = Trainer(
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cfg=cfg,
                    run_dir=run_dir,
                    artifact_writer=writer,
                    callbacks=cb_list,
                )

            logger.info("[train] dl_name=%s batch_size=%s num_workers=%s", dl_name, cfg.data.batch_size, cfg.data.num_workers)
            logger.info("[train] model=%s loss=%s optim=%s sched=%s", cfg.model.name, cfg.loss.name, cfg.optim.name, cfg.sched.name)

            best_ckpt = trainer.fit(train_loader=train_loader, val_loader=val_loader)

            logger.info("[train] best_checkpoint=%s", best_ckpt)
            logger.info("[train] split %d done", split_index + 1)

        # Log suggestion for prediction
        parent_run_dir = os.path.join(cfg.save_dir, cfg.run_name, stamp)
        logger.info("[train] Training completed for all splits under run directory: %s", parent_run_dir)
        logger.info("[train] To run prediction on all splits, execute:")
        logger.info("  seizure-pred predict --config %s --checkpoint %s", args.config, parent_run_dir)
    finally:
        try:
            logger = logging.getLogger("seizure_pred")
            for h in list(logger.handlers):
                h.close()
                logger.removeHandler(h)
        except Exception:
            pass


def train_from_config(
    config_path: os.PathLike[str] | str,
    *,
    dataloader: str | None = None,
    mil: bool = False,
    override_path: os.PathLike[str] | str | None = None,
    strict: bool = False,
) -> None:
    """Programmatic entrypoint used by legacy scripts.

    This mirrors the CLI behavior of ``seizure-pred train`` but is convenient for
    back-compat wrappers that historically imported a function.
    """

    ns = argparse.Namespace(
        config=str(config_path),
        override=None if override_path is None else str(override_path),
        n_folds=5,
        dataloader=dataloader,
        mil=bool(mil),
        strict=bool(strict),
        print_config=False,
        outer_method=None,
        inner_method=None,
        outer_n_fold=None,
        inner_n_fold=None,
        outer_shuffle=None,
        inner_shuffle=None,
    )
    run_train(ns)


def ensemble_outer_split(outer_split_dir: str, inner_n_fold: int, threshold: float = 0.5) -> None:
    import json
    import os
    import numpy as np
    import torch
    from seizure_pred.training.engine.artifacts import ArtifactWriter
    from seizure_pred.training.engine.metrics import binary_classification_metrics

    inner_probs = []
    val_aucs = []
    y_true = []
    metas = []

    for inner_idx in range(inner_n_fold):
        inner_dir = os.path.join(outer_split_dir, f"inner_split_{inner_idx}")
        
        # Read val metric (AUC)
        val_metrics_path = os.path.join(inner_dir, "val_metrics.json")
        val_auc = 0.5
        if os.path.exists(val_metrics_path):
            try:
                with open(val_metrics_path, "r", encoding="utf-8") as f:
                    val_data = json.load(f)
                    val_auc = val_data.get("metrics", {}).get("auc", 0.5)
                    if val_auc is None or np.isnan(val_auc):
                        val_auc = 0.5
            except Exception:
                pass
        val_aucs.append(val_auc)

        # Read predictions
        preds_path = os.path.join(inner_dir, "predictions.jsonl")
        probs = []
        if os.path.exists(preds_path):
            try:
                with open(preds_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if "prob" in row:
                            probs.append(float(row["prob"]))
                        elif "logit" in row:
                            logit = float(row["logit"])
                            probs.append(1.0 / (1.0 + np.exp(-logit)))
                        else:
                            probs.append(0.5)
                        if inner_idx == 0:
                            yt = row.get("y_true", row.get("target", 0))
                            y_true.append(int(yt))
                            metas.append(row.get("meta"))
            except Exception:
                pass
        inner_probs.append(probs)

    if not y_true or not inner_probs:
        return

    inner_probs = np.array(inner_probs)  # shape (inner_n_fold, num_samples)
    val_aucs = np.array(val_aucs)
    
    if np.sum(val_aucs) > 0:
        weights = val_aucs / np.sum(val_aucs)
    else:
        weights = np.ones_like(val_aucs) / len(val_aucs)

    # Weighted sum along axis 0
    ensembled_probs = np.tensordot(weights, inner_probs, axes=1)

    # Write ensembled predictions to outer_split_dir/predictions.jsonl
    out_writer = ArtifactWriter(outer_split_dir)
    rows = []
    for i in range(len(y_true)):
        prob = float(ensembled_probs[i])
        y_pred = int(prob >= threshold)
        rows.append({
            "y_true": int(y_true[i]),
            "prob": prob,
            "y_pred": y_pred,
            "meta": metas[i]
        })
    out_writer.write_predictions(rows)

    # Compute ensembled metrics and write to outer_split_dir/metrics.json
    probs_tensor = torch.tensor(ensembled_probs)
    # Convert probability to logit safely: logit = log(p / (1 - p))
    logits_ensemble = torch.log(probs_tensor / torch.clamp(1.0 - probs_tensor, min=1e-8))
    targets_tensor = torch.tensor(y_true)
    metrics = binary_classification_metrics(logits_ensemble, targets_tensor, threshold=threshold)
    out_writer.write_metrics(metrics)


def run_nested_cv(cfg: TrainConfig, dataset: Any, stamp: str, args: argparse.Namespace) -> None:
    import pickle
    import numpy as np
    import torch
    from seizure_pred.data.splits import make_cv_splitter
    from seizure_pred.training.engine.checkpoint import restore_checkpoint
    
    dl_name = cfg.data.dataloader_type or "torch"
    run_root_dir = os.path.join(cfg.save_dir, cfg.run_name, stamp)
    os.makedirs(run_root_dir, exist_ok=True)
    logger = setup_logging(run_root_dir)
    
    # Outer CV splits
    outer_splits = list(make_cv_splitter(
        dataset,
        method=cfg.cv.outer_method,
        shuffle=cfg.cv.outer_shuffle,
        random_state=cfg.cv.random_state,
        n_fold=cfg.cv.outer_n_fold,
        mode=cfg.cv.outer_mode,
        M=cfg.cv.outer_M,
    ))
    
    if not outer_splits:
        raise SystemExit("No outer splits found to train on.")
        
    cv_results = {'outer_folds': []}
    
    try:
        for outer_idx, (train_val_set, test_set) in enumerate(outer_splits):
            logger.info("\n" + "=" * 80)
            logger.info("OUTER FOLD %d/%d", outer_idx + 1, len(outer_splits))
            logger.info("=" * 80)
            
            # Setup outer fold tracking data
            test_y = []
            test_indices = []
            if hasattr(test_set, "y"):
                test_y = test_set.y.tolist()
            if hasattr(test_set, "base_indices"):
                test_indices = test_set.base_indices.tolist()
                
            fold_data = {
                'outer_fold': outer_idx + 1,
                'test_indices': test_indices,
                'y_test': test_y,
                'inner_folds': []
            }
            
            # Inner CV splits
            inner_splits = list(make_cv_splitter(
                train_val_set,
                method=cfg.cv.inner_method,
                n_fold=cfg.cv.inner_n_fold,
                shuffle=cfg.cv.inner_shuffle,
                mode=cfg.cv.inner_mode,
                M=cfg.cv.inner_M,
                random_state=cfg.cv.random_state,
            ))
            
            for inner_idx, (train_set, val_set) in enumerate(inner_splits):
                logger.info("\n--- Inner Fold %d/%d ---", inner_idx + 1, len(inner_splits))
                logger.info("Train: %d samples", len(train_set))
                logger.info("Val: %d samples", len(val_set))
                logger.info("Test: %d samples", len(test_set))
                
                # Determinism / seeding
                seed_everything(getattr(cfg, "determinism", None), seed=cfg.seed)
                
                # Setup directories
                inner_dir = os.path.join(run_root_dir, f"split_{outer_idx}", f"inner_split_{inner_idx}")
                os.makedirs(inner_dir, exist_ok=True)
                
                # Setup logging
                inner_logger = setup_logging(inner_dir)
                
                # Build loaders
                train_loader = build_loader(dl_name, train_set, cfg, shuffle=True)
                val_loader = build_loader(dl_name, val_set, cfg, shuffle=False)
                test_loader = build_loader(dl_name, test_set, cfg, shuffle=False)
                
                # Build components
                model = MODELS.create(cfg.model.name, cfg.model)
                loss_fn = LOSSES.create(cfg.loss.name, **(cfg.loss.kwargs or {}))
                optimizer = OPTIMIZERS.create(
                    cfg.optim.name,
                    model.parameters(),
                    lr=cfg.optim.lr,
                    weight_decay=cfg.optim.weight_decay,
                    **(cfg.optim.kwargs or {}),
                )
                scheduler = None
                if cfg.sched.name:
                    scheduler = SCHEDULERS.create(cfg.sched.name, optimizer, **(cfg.sched.kwargs or {}))
                    
                cb_list = []
                for item in getattr(cfg, "callbacks", []) or []:
                    name = item["name"] if isinstance(item, dict) else getattr(item, "name", None)
                    kwargs = item.get("kwargs", {}) if isinstance(item, dict) else dict(getattr(item, "kwargs", {}) or {})
                    if not name:
                        continue
                    cb_list.append(CALLBACKS.create(name, **kwargs))
                    
                writer = ArtifactWriter(inner_dir)
                writer.write_schema()
                writer.write_config(asdict(cfg))
                
                # Instantiate appropriate trainer
                if args.mil:
                    trainer = TrainerMIL(
                        model=model,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        cfg=cfg,
                        run_dir=inner_dir,
                        artifact_writer=writer,
                        callbacks=cb_list,
                    )
                else:
                    trainer = Trainer(
                        model=model,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        cfg=cfg,
                        run_dir=inner_dir,
                        artifact_writer=writer,
                        callbacks=cb_list,
                    )
                    
                # Train
                best_ckpt = trainer.fit(train_loader=train_loader, val_loader=val_loader, write_best_predictions=False)
                
                # Evaluate best model
                restore_checkpoint(best_ckpt, model=model)
                val_out = trainer.evaluate(val_loader)
                test_out = trainer.evaluate(test_loader)
                
                # Convert evaluation outputs to metrics/predictions
                val_metrics = {k: v for k, v in val_out.items() if k not in {"val_logits", "val_targets", "val_meta"}}
                test_metrics = {k: v for k, v in test_out.items() if k not in {"val_logits", "val_targets", "val_meta"}}
                
                val_probs = torch.sigmoid(val_out["val_logits"]).tolist()
                val_preds = (torch.sigmoid(val_out["val_logits"]) >= 0.5).long().tolist()
                
                test_probs = torch.sigmoid(test_out["val_logits"]).tolist()
                test_preds = (torch.sigmoid(test_out["val_logits"]) >= 0.5).long().tolist()
                
                # Write files for this inner split
                writer.write_metrics(val_metrics, filename="val_metrics.json")
                writer.write_metrics(test_metrics, filename="test_metrics.json")
                # Convert logits, targets, and meta into row dicts to ensure standard predictions format
                test_probs_tensor = torch.sigmoid(test_out["val_logits"])
                test_preds_tensor = (test_probs_tensor >= 0.5).long()
                test_rows = []
                for idx_sample in range(len(test_probs_tensor)):
                    test_rows.append({
                        "y_true": int(test_out["val_targets"][idx_sample].item()),
                        "logit": float(test_out["val_logits"][idx_sample].item()),
                        "prob": float(test_probs_tensor[idx_sample].item()),
                        "y_pred": int(test_preds_tensor[idx_sample].item()),
                        "meta": test_out["val_meta"][idx_sample] if idx_sample < len(test_out["val_meta"]) else None
                    })
                writer.write_predictions(test_rows)
                
                # Load training history to match legacy log format
                history_rows = []
                history_path = os.path.join(inner_dir, "history.jsonl")
                if os.path.exists(history_path):
                    with open(history_path, "r", encoding="utf-8") as h_f:
                        for line in h_f:
                            if line.strip():
                                history_rows.append(json.loads(line))
                                
                val_indices = val_set.base_indices.tolist() if hasattr(val_set, "base_indices") else list(range(len(val_set)))
                val_labels = val_set.y.tolist() if hasattr(val_set, "y") else [int(y) for y in val_out["val_targets"]]
                test_labels = test_set.y.tolist() if hasattr(test_set, "y") else [int(y) for y in test_out["val_targets"]]
                
                inner_fold_data = {
                    'inner_fold': inner_idx + 1,
                    'best_val_auc': val_metrics.get("auc", 0.5),
                    'model_path': best_ckpt,
                    'training_log': history_rows,
                    'val_indices': val_indices,
                    'val_probs': val_probs,
                    'val_preds': val_preds,
                    'val_labels': val_labels,
                    'test_probs': test_probs,
                    'test_preds': test_preds,
                    'test_labels': test_labels,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics
                }
                fold_data['inner_folds'].append(inner_fold_data)
                
                # Clean up log handlers for the inner fold
                try:
                    for h in list(inner_logger.handlers):
                        h.close()
                        inner_logger.removeHandler(h)
                except Exception:
                    pass
                    
            cv_results['outer_folds'].append(fold_data)
            
            # Perform ensembled prediction for the outer fold
            outer_split_dir = os.path.join(run_root_dir, f"split_{outer_idx}")
            ensemble_outer_split(outer_split_dir, len(inner_splits), threshold=args.threshold if hasattr(args, "threshold") else 0.5)
            logger.info("Outer Fold %d Complete. Ensembled test predictions written to %s", outer_idx + 1, outer_split_dir)
            
        # Write legacy raw_predictions.pkl
        results_path = os.path.join(run_root_dir, 'raw_predictions.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(cv_results, f)
            
        logger.info("\n" + "=" * 80)
        logger.info("✨ NESTED TRAINING COMPLETE!")
        logger.info("=" * 80)
        logger.info("📁 Results saved to: %s", run_root_dir)
        logger.info("🎯 Out-of-fold test predictions and metrics have been written directly to:")
        logger.info("  %s/split_*/predictions.jsonl", run_root_dir)
        logger.info("🎯 To run analysis on these predictions, execute:")
        logger.info("  seizure-pred analyze --run-dir %s", run_root_dir)
        logger.info("🎯 To predict on a completely new dataset using these trained models, execute:")
        logger.info("  seizure-pred predict --config %s --checkpoint %s", args.config, run_root_dir)
        logger.info("🎯 Run analysis (legacy): python E:\\Projects\\seizure\\library\\CP-DMGC-CWT\\analyze_results3.py --run_dir %s", run_root_dir)
        logger.info("=" * 80 + "\n")
        
    finally:
        try:
            for h in list(logger.handlers):
                h.close()
                logger.removeHandler(h)
        except Exception:
            pass
