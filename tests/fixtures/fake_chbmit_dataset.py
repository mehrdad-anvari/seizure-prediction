from pathlib import Path
import pandas as pd


def create_fake_chbmit_dataset(tmp_path: Path):

    eeg_dir = (
        tmp_path
        / "sub-01"
        / "ses-01"
        / "eeg"
    )

    eeg_dir.mkdir(parents=True, exist_ok=True)

    # fake EDF file
    (eeg_dir / "fake_run-00_eeg.edf").touch()

    # fake annotation TSV
    df = pd.DataFrame({
        "onset": [1000.0],
        "duration": [40.0],
        "eventType": ["sz"],
        "confidence": ["n/a"],
        "channels": ["n/a"],
        "dateTime": ["2076-11-06 13:43:04"],
        "recordingDuration": [3600.0],
    })

    df.to_csv(
        eeg_dir / "fake_run-00_events.tsv",
        sep="\t",
        index=False,
    )

    return tmp_path