from seizure_pred.preprocessing.chbmit_bids import process_chbmit_bids_dataset

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
 