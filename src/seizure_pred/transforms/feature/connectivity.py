import warnings
import numpy as np

from .base import FeatureTransform


def _require_mne_connectivity():
    """Import mne + mne-connectivity lazily with a helpful error message."""
    try:
        import mne  # noqa: F401
    except Exception as e:
        raise ImportError(
            "This transform requires `mne`. Install with: pip install seizure-pred[eeg]"
        ) from e
    try:
        from mne_connectivity import spectral_connectivity_epochs  # type: ignore
    except Exception as e:
        raise ImportError(
            "This transform requires `mne-connectivity`. Install with: pip install seizure-pred[eeg]"
        ) from e
    return spectral_connectivity_epochs


class MNEConnectivityBase(FeatureTransform):
    """Base class for MNE connectivity measures using spectral_connectivity_epochs."""

    def __init__(
        self,
        method: str = "coh",
        fmin: float = 0.5,
        fmax: float = 50.0,
        sfreq: float = 128.0,
        mode: str = "multitaper",
        faverage: bool = True,
    ):
        self._spectral_connectivity_epochs = _require_mne_connectivity()
        self.method = method
        self.fmin = fmin
        self.fmax = fmax
        self.sfreq = sfreq
        self.mode = mode
        self.faverage = faverage

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        n_channels, n_times = eeg.shape
        if n_channels <= 1:
            return np.full((n_channels, 1), np.nan)

        data_mne = eeg[np.newaxis, :, :]  # (1, n_channels, n_times)

        mean_con = np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                con = self._spectral_connectivity_epochs(
                    data_mne,
                    method=self.method,
                    mode=self.mode,
                    sfreq=self.sfreq,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    faverage=self.faverage,
                    n_jobs=1,
                    verbose=False,
                )

            con_matrix = con.get_data(output="dense").squeeze()

            if (
                isinstance(con_matrix, np.ndarray)
                and con_matrix.ndim == 2
                and con_matrix.shape[0] == n_channels
                and con_matrix.shape[1] == n_channels
            ):
                iu = np.triu_indices(n_channels, k=1)
                con_values = con_matrix[iu]
                if np.any(np.isfinite(con_values)):
                    mean_con = float(np.nanmean(np.abs(con_values)))
        except Exception:
            # If MNE errors for a segment, keep NaN.
            mean_con = np.nan

        return np.full((n_channels, 1), mean_con, dtype=float)


class MeanAbsCorrelation(FeatureTransform):
    """Mean absolute Pearson correlation between all channel pairs."""

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        n_channels = eeg.shape[0]
        if n_channels <= 1:
            return np.full((n_channels, 1), np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = np.corrcoef(eeg)

        if not np.all(np.isfinite(corr)):
            return np.full((n_channels, 1), np.nan)

        iu = np.triu_indices(n_channels, k=1)
        val = float(np.mean(np.abs(corr[iu]))) if len(corr[iu]) else np.nan
        return np.full((n_channels, 1), val)


class MeanCoh(MNEConnectivityBase):
    """Mean coherence across all channel pairs."""

    def __init__(self, fmin=0.5, fmax=50, sfreq=128):
        super().__init__(method="coh", fmin=fmin, fmax=fmax, sfreq=sfreq)


class MeanPLV(MNEConnectivityBase):
    """Mean Phase Locking Value across all channel pairs."""

    def __init__(self, fmin=0.5, fmax=50, sfreq=128):
        super().__init__(method="plv", fmin=fmin, fmax=fmax, sfreq=sfreq)


class MeanImCoh(MNEConnectivityBase):
    """Mean imaginary coherence across all channel pairs."""

    def __init__(self, fmin=0.5, fmax=50, sfreq=128):
        super().__init__(method="imcoh", fmin=fmin, fmax=fmax, sfreq=sfreq)


class MeanPLI(MNEConnectivityBase):
    """Mean Phase Lag Index across all channel pairs."""

    def __init__(self, fmin=0.5, fmax=50, sfreq=128):
        super().__init__(method="pli", fmin=fmin, fmax=fmax, sfreq=sfreq)


class MeanWPLI(MNEConnectivityBase):
    """Mean Weighted Phase Lag Index across all channel pairs."""

    def __init__(self, fmin=0.5, fmax=50, sfreq=128):
        super().__init__(method="wpli", fmin=fmin, fmax=fmax, sfreq=sfreq)
