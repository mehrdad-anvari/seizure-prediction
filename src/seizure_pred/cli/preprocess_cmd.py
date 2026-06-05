from __future__ import annotations

import argparse


def add_preprocess_cmd(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("preprocess-chbmit", help="Preprocess CHB-MIT BIDS -> NPZ sessions")

    p.add_argument("--dataset-dir", required=True, help="Path to BIDS_CHB-MIT root")
    p.add_argument("--subject", default=None, help="Subject number(s) like 1,2,3 or 'all'")
    p.add_argument("--save-uint16", action="store_true")
    p.add_argument("--no-filter", action="store_true")
    p.add_argument("--no-ica", action="store_true")
    p.add_argument("--no-downsample", action="store_true")

    p.add_argument("--filter-type", default="FIR", choices=["FIR", "IIR"])
    p.add_argument("--l-freq", type=float, default=0.5)
    p.add_argument("--h-freq", type=float, default=50.0)

    p.add_argument("--sfreq-new", type=float, default=128.0)
    p.add_argument("--downsample-method", default="polyphase", choices=["polyphase", "fft", "resample"])

    p.add_argument("--normalize", default=None, choices=[None, "zscore", "robust"], nargs="?")

    p.add_argument("--preictal-minutes", type=int, default=15)
    p.add_argument("--pre-buffer-minutes", type=int, default=45)
    p.add_argument("--post-buffer-minutes", type=int, default=60)

    p.add_argument("--segment-sec", type=int, default=5)
    p.add_argument("--preictal-oversample-factor", type=int, default=1)
    p.add_argument("--seizure-oversample-factor", type=int, default=1)

    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-psd", action="store_true")

    p.set_defaults(func=run_preprocess_cmd)


def run_preprocess_cmd(args: argparse.Namespace) -> None:
    from seizure_pred.preprocessing.chbmit_bids import process_chbmit_bids_dataset

    subj_nums = None
    if args.subject and str(args.subject).lower() not in {"all", "*"}:
        subj_nums = [int(s.strip()) for s in str(args.subject).split(",") if s.strip()]

    process_chbmit_bids_dataset(
        dataset_dir=args.dataset_dir,
        save_uint16=args.save_uint16,
        apply_filter=not args.no_filter,
        apply_ica=not args.no_ica,
        apply_downsampling=not args.no_downsample,
        filter_type=args.filter_type,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        sfreq_new=args.sfreq_new,
        downsample_method=args.downsample_method,
        normalize=args.normalize,
        plot=args.plot,
        plot_psd=args.plot_psd,
        subj_nums=subj_nums,
        preictal_oversample_factor=args.preictal_oversample_factor,
        seizure_oversample_factor=args.seizure_oversample_factor,
        preictal_minutes=args.preictal_minutes,
        post_buffer_minutes=args.post_buffer_minutes,
        pre_buffer_minutes=args.pre_buffer_minutes,
        segment_sec=args.segment_sec,
    )
