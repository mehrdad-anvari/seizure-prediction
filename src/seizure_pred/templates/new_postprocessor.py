
from __future__ import annotations

from seizure_pred.training.registries import POSTPROCESSORS
from seizure_pred.inference.postprocess import Postprocessor, PostprocessResult
import numpy as np

@POSTPROCESSORS.register("my_postprocessor", help="Describe what it does.")
class MyPostprocessor(Postprocessor):
    def __init__(self, **kwargs):
        ...

    def __call__(self, probs, meta=None) -> PostprocessResult:
        p = np.asarray(probs, dtype=np.float32)
        # do something
        pred = (p >= 0.5).astype(np.int64)
        return PostprocessResult(probs=p, pred=pred)
