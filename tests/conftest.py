import pytest
import mne
from .fixtures.fake_mne import make_fake_raw
from .fixtures.fake_chbmit_dataset import create_fake_chbmit_dataset

@pytest.fixture
def fake_read_raw_edf(monkeypatch):

    def _fake_read_raw_edf(*args, **kwargs):
        return make_fake_raw()

    monkeypatch.setattr(
        mne.io,
        "read_raw_edf",
        _fake_read_raw_edf,
    )

@pytest.fixture
def fake_dataset(tmp_path):
    return create_fake_chbmit_dataset(tmp_path)

@pytest.fixture
def fake_raw():
    return make_fake_raw()