
from __future__ import annotations

from seizure_pred.training.registries import POSTPROCESSORS
from seizure_pred.inference.postprocess import Threshold, MovingAverage, Hysteresis, Compose

@POSTPROCESSORS.register("threshold", help="Simple fixed threshold on probs.")
def build_threshold(threshold: float = 0.5, **kwargs):
    return Threshold(threshold=threshold)

@POSTPROCESSORS.register("moving_average", help="Moving average smoothing + threshold.")
def build_moving_average(window: int = 5, threshold: float = 0.5, centered: bool = False, **kwargs):
    return MovingAverage(window=window, threshold=threshold, centered=centered)

@POSTPROCESSORS.register("hysteresis", help="Hysteresis thresholds with optional smoothing and min on/off.")
def build_hysteresis(
    on_threshold: float = 0.6,
    off_threshold: float = 0.4,
    min_on: int = 1,
    min_off: int = 1,
    smoothing_window: int = 1,
    smoothing_centered: bool = False,
    **kwargs
):
    return Hysteresis(
        on_threshold=on_threshold,
        off_threshold=off_threshold,
        min_on=min_on,
        min_off=min_off,
        smoothing_window=smoothing_window,
        smoothing_centered=smoothing_centered,
    )

@POSTPROCESSORS.register("compose", help="Compose multiple postprocessors; provide 'steps' list of dicts.")
def build_compose(steps, **kwargs):
    # steps: [{"name":"moving_average","kwargs":{...}}, {"name":"threshold","kwargs":{...}}]
    from seizure_pred.training.registries import POSTPROCESSORS as REG
    built = []
    for s in steps:
        name = s["name"]
        kw = s.get("kwargs", {}) or {}
        built.append(REG.create(name, **kw))
    return Compose(built)
