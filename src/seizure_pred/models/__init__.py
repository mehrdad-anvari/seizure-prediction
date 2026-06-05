from __future__ import annotations

# Keep light imports by default.
from . import api  # noqa: F401
from . import simple_cnn  # noqa: F401

from seizure_pred.core.optional_deps import is_torch_geometric_available


def _try_import(mod: str) -> bool:
    try:
        __import__(mod, fromlist=["*"])
        return True
    except Exception:
        return False


def register_all() -> None:
    """Import model zoo modules so they register with MODELS.

    Some models have optional dependencies (e.g., torch_geometric, scipy). Those
    are imported best-effort and will be skipped if dependencies are missing.
    """
    # Classic CNN/EEG models (torch-only)
    from . import eegnet  # noqa: F401
    from . import eegwavenet  # noqa: F401
    from . import tsception  # noqa: F401
    _try_import("seizure_pred.models.fbmsnet")
    from . import lmda  # noqa: F401
    from . import tslanet  # noqa: F401
    from . import simplevit  # noqa: F401
    from . import eegbandclassifier  # noqa: F401

    # Additional models ported from the original repository (torch-only)
    from . import mb_dmgc_cwtffnet  # noqa: F401

    # Optional: LaBraM (requires einops)
    _try_import("seizure_pred.models.labram")

    # Conformer package + wrapper (torch-only)
    from . import conformer_model  # noqa: F401

    # Torch-only graph-ish models (no torch_geometric)
    _try_import("seizure_pred.models.cspnet")
    _try_import("seizure_pred.models.dgcnn")

    # Optional: GNN/graph models (torch_geometric)
    if is_torch_geometric_available():
        _try_import("seizure_pred.models.dgcnn2")
        _try_import("seizure_pred.models.rgnn")

    # Optional: CE-stSENet (scipy for .mat filters)
    _try_import("seizure_pred.models.ce_stsenet.ce_stsenet")

    # Optional: EEG-GNN-SSL / DCRNN graph model (scipy/mne for adjacency unless provided)
    _try_import("seizure_pred.models.eeg_gnn_ssl")

    # Optional: STNet (depends on optional packages in some forks; keep best-effort)
    _try_import("seizure_pred.models.stnet")
