from seizure_pred.preprocessing.chbmit_bids import process_chbmit_bids_dataset
from seizure_pred.preprocessing.utils import extract_segments_with_labels_bids


def test_preprocess_cli_accepts_interictal_oversample_factor():
    from seizure_pred.cli.main import build_parser

    args = build_parser().parse_args([
        "preprocess-chbmit",
        "--dataset-dir", "dataset",
        "--interictal-oversample-factor", "3",
    ])

    assert args.interictal_oversample_factor == 3


def test_interictal_oversampling_increases_segments(fake_raw):
    import mne

    fake_raw.set_annotations(mne.Annotations(
        onset=[0], duration=[20], description=["interictal"]
    ))

    baseline = extract_segments_with_labels_bids(
        fake_raw, segment_sec=5, interictal_oversample_factor=1
    )
    oversampled = extract_segments_with_labels_bids(
        fake_raw, segment_sec=5, interictal_oversample_factor=3
    )

    assert len(oversampled[1]) > len(baseline[1])
    assert set(oversampled[3][0].keys()) >= {"applied_factor", "applied_overlap_sec"}
    assert oversampled[3][0]["applied_factor"] == 3.0
    assert oversampled[2]["augmented"].tolist() == [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]

def test_smoke_preprocessing_pipeline(fake_dataset, fake_read_raw_edf, monkeypatch):

    process_chbmit_bids_dataset(
        dataset_dir=str(fake_dataset),
        apply_filter=False,
        apply_ica=False,
        apply_downsampling=False,
        normalize=None,
        show_statistics=False,
        plot=False,
        plot_psd=False,
        save_uint16=False,
        segment_sec=5,
        preictal_minutes=1,
        post_buffer_minutes=1,
        pre_buffer_minutes=1,
    )

    session_dir = (
        fake_dataset
        / "sub-01"
        / "ses-01"
        / "eeg"
    )

    assert (session_dir / "event_stats.csv").exists()
    assert len(list(session_dir.glob("*.npz"))) > 0
