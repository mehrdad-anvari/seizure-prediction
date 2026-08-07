from __future__ import annotations

import argparse

from seizure_pred.analysis.runner import analyze_run


def add_analyze_cmd(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("analyze", help="Analyze a run and write plots/reports")
    p.add_argument("--run-dir", required=True, help="Run directory containing predictions/history")
    p.add_argument("--out-dir", default=None, help="Output directory (default: <run_dir>/analysis)")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for y_pred if needed")
    p.add_argument("--prefer-postprocessed", action="store_true", help="Use y_pred_post if present")
    p.add_argument("--no-plots", action="store_true", help="Skip writing plots (CI-friendly)")
    p.add_argument("--sampling-period", type=float, default=5.0, help="Duration of each prediction sample in seconds")

    # Calibration sweep (nested-CV runs with raw_predictions.pkl)
    p.add_argument("--calibration-methods", nargs="+", default=None,
                   help="Calibration methods to sweep (nested CV runs). "
                        "Choices: none percentile beta isotonic temperature. Default: all.")
    p.add_argument("--ma-windows", nargs="+", type=int, default=[1, 3, 5, 7, 10],
                   help="Moving-average windows to sweep (default: 1 3 5 7 10)")
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7],
                   help="Thresholds to sweep (default: 0.3 0.4 0.5 0.6 0.7)")
    p.add_argument("--percentiles", nargs="+", type=int, default=None,
                   help="Percentiles for percentile calibration (default: 5 10 15 20)")
    p.add_argument("--suppression-duration", type=int, default=None,
                   help="Suppression window (in samples) for the suppressed FPR/hour metric")
    p.set_defaults(func=run_analyze_cmd)


def run_analyze_cmd(args: argparse.Namespace) -> None:
    import os
    from seizure_pred.core.runs import find_splits

    # Tolerate minimal Namespaces constructed programmatically (e.g. in tests).
    ma_windows = getattr(args, "ma_windows", None)
    thresholds = getattr(args, "thresholds", None)
    thresholds_list = [float(t) for t in thresholds] if thresholds else None
    calibration_methods = getattr(args, "calibration_methods", None)
    percentiles = getattr(args, "percentiles", None)
    suppression_duration = getattr(args, "suppression_duration", None)
    sampling_period = getattr(args, "sampling_period", 5.0)
    no_plots = getattr(args, "no_plots", False)

    if not no_plots:
        from seizure_pred.analysis.nested_predictions import (
            analyze_interictal_prob,
            analyze_interictal_pp_scatter,
            analyze_preictal_prob,
            analyze_pp_scatter_combined,
        )
        probability_analyses = (
            ("preictal", analyze_preictal_prob),
            ("interictal", analyze_interictal_prob),
        )

    split_dirs = find_splits(args.run_dir)
    if split_dirs:
        for split_index, split_dir in split_dirs:
            split_out_dir = None
            if args.out_dir is not None:
                split_out_dir = os.path.join(args.out_dir, f"split_{split_index}")
            analyze_run(
                run_dir=split_dir,
                out_dir=split_out_dir,
                threshold=args.threshold,
                prefer_postprocessed=args.prefer_postprocessed,
                make_plots=not no_plots,
            )
            if not no_plots:
                for event_type, analyzer in probability_analyses:
                    try:
                        comparison = analyzer(
                            split_dir,
                            out_dir=split_out_dir,
                            sampling_period=sampling_period,
                        )
                        if comparison["status"] == "ok":
                            print(
                                f"[analysis] Generated nested {event_type} "
                                f"comparison for split_{split_index}"
                            )
                    except Exception as e:
                        print(
                            f"[analysis] Failed nested {event_type} comparison "
                            f"for split_{split_index}: {e}"
                        )
                # P-P scatter (interictal only, colour by epoch index).
                try:
                    scatter_result = analyze_interictal_pp_scatter(
                        split_dir,
                        out_dir=split_out_dir,
                        sampling_period=sampling_period,
                    )
                    if scatter_result["status"] == "ok":
                        print(
                            f"[analysis] Generated interictal P-P scatter "
                            f"for split_{split_index}"
                        )
                except Exception as e:
                    print(
                        f"[analysis] Failed interictal P-P scatter "
                        f"for split_{split_index}: {e}"
                    )
                # Combined P-P scatter (interictal + preictal, colour-coded).
                try:
                    combined = analyze_pp_scatter_combined(
                        split_dir,
                        out_dir=split_out_dir,
                        sampling_period=sampling_period,
                    )
                    if combined["status"] == "ok":
                        print(
                            f"[analysis] Generated combined P-P scatter "
                            f"for split_{split_index}"
                        )
                except Exception as e:
                    print(
                        f"[analysis] Failed combined P-P scatter "
                        f"for split_{split_index}: {e}"
                    )

        # Per-split MA x threshold sweep + Pareto (works on predictions.jsonl)
        from seizure_pred.analysis.summary import analyze_multi_split_summary
        analyze_multi_split_summary(
            run_dir=args.run_dir,
            out_dir=args.out_dir,
            ma_windows=ma_windows,
            thresholds=thresholds_list,
            sampling_period=sampling_period,
            make_plots=not no_plots,
        )

        # Calibration x MA x threshold sweep (requires raw_predictions.pkl from nested CV)
        if os.path.exists(os.path.join(args.run_dir, "raw_predictions.pkl")):
            from seizure_pred.analysis.calibration_sweep import analyze_nested_calibration
            analyze_nested_calibration(
                run_dir=args.run_dir,
                out_dir=args.out_dir,
                calibration_methods=calibration_methods,
                ma_windows=ma_windows,
                thresholds=thresholds_list,
                percentiles=percentiles,
                sampling_period=sampling_period,
                suppression_duration=suppression_duration,
                make_plots=not no_plots,
            )
    else:
        analyze_run(
            run_dir=args.run_dir,
            out_dir=args.out_dir,
            threshold=args.threshold,
            prefer_postprocessed=args.prefer_postprocessed,
            make_plots=not no_plots,
        )
        if not no_plots:
            split_name = os.path.basename(os.path.normpath(args.run_dir))
            for event_type, analyzer in probability_analyses:
                try:
                    comparison = analyzer(
                        args.run_dir,
                        out_dir=args.out_dir,
                        sampling_period=sampling_period,
                    )
                    if comparison["status"] == "ok":
                        print(
                            f"[analysis] Generated nested {event_type} "
                            f"comparison for {split_name}"
                        )
                except Exception as e:
                    print(
                        f"[analysis] Failed nested {event_type} comparison "
                        f"for {split_name}: {e}"
                    )
            # P-P scatter (interictal only, colour by epoch index).
            try:
                scatter_result = analyze_interictal_pp_scatter(
                    args.run_dir,
                    out_dir=args.out_dir,
                    sampling_period=sampling_period,
                )
                if scatter_result["status"] == "ok":
                    print(
                        f"[analysis] Generated interictal P-P scatter "
                        f"for {split_name}"
                    )
            except Exception as e:
                print(
                    f"[analysis] Failed interictal P-P scatter "
                    f"for {split_name}: {e}"
                )
            # Combined P-P scatter (interictal + preictal, colour-coded).
            try:
                combined = analyze_pp_scatter_combined(
                    args.run_dir,
                    out_dir=args.out_dir,
                    sampling_period=sampling_period,
                )
                if combined["status"] == "ok":
                    print(
                        f"[analysis] Generated combined P-P scatter "
                        f"for {split_name}"
                    )
            except Exception as e:
                print(
                    f"[analysis] Failed combined P-P scatter "
                    f"for {split_name}: {e}"
                )
