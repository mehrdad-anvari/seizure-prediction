import pytest


def _require_mne():
    """Return the mne module or skip the test (mne is an optional extra)."""
    try:
        import mne  # noqa: F401
        return mne
    except Exception:
        pytest.skip("mne not installed (install with: pip install seizure-pred[eeg])")


@pytest.fixture
def fake_read_raw_edf(monkeypatch):
    mne = _require_mne()

    def _fake_read_raw_edf(*args, **kwargs):
        from .fixtures.fake_mne import make_fake_raw
        return make_fake_raw()

    monkeypatch.setattr(mne.io, "read_raw_edf", _fake_read_raw_edf)


@pytest.fixture
def fake_dataset(tmp_path):
    _require_mne()
    from .fixtures.fake_chbmit_dataset import create_fake_chbmit_dataset
    return create_fake_chbmit_dataset(tmp_path)


@pytest.fixture
def fake_raw():
    _require_mne()
    from .fixtures.fake_mne import make_fake_raw
    return make_fake_raw()
