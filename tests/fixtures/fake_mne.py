import numpy as np
import mne

def make_fake_raw(n_channels=3, duration_sec=3600, sfreq=128, seed=42):
    rng = np.random.RandomState(seed)

    n_times = int(duration_sec * sfreq)

    data = rng.randn(n_channels, n_times)

    info = mne.create_info(
        ch_names=[f"C{i}" for i in range(n_channels)],
        sfreq=sfreq,
        ch_types="eeg"
    )

    return mne.io.RawArray(data, info)