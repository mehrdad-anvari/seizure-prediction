"""Probability calibration for seizure prediction.

Ported from the original repository's ``probability_calibration.py`` and made
dependency-tolerant: ``expit``/``logit`` are implemented locally so that the
``percentile`` method needs no SciPy; ``beta``/``temperature`` require
``scipy.optimize.minimize`` and ``isotonic`` requires scikit-learn. Missing
optional dependencies raise a clear ``ImportError`` only when the corresponding
method is actually used.

Supported methods
-----------------
- ``percentile``  : sigmoid ``expit(a*logit(p)+b)`` mapping a target percentile
  of preictal validation probabilities to 0.5 (reduces false positives).
- ``beta``        : 3-parameter beta calibration ``expit(a + b*log(p) + c*log(1-p))``.
- ``isotonic``    : non-parametric monotonic regression (scikit-learn).
- ``temperature`` : temperature scaling with bias ``expit((logit(p)-b)/T)``.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("seizure_pred")


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def _require(module: str, hint: str):
    try:
        return __import__(module, fromlist=["*"])
    except Exception as e:  # pragma: no cover
        raise ImportError(
            f"Calibration method needs optional dependency '{module}'. "
            f"Install with: pip install {hint}"
        ) from e


class ProbabilityCalibrator:
    """Fit-on-validation / transform-on-test probability calibrator.

    Parameters
    ----------
    method:
        One of ``"percentile"``, ``"beta"``, ``"isotonic"``, ``"temperature"``.
    target_preictal_percentile:
        Used by the ``percentile`` method: the bottom-N% of preictal validation
        probabilities are mapped below 0.5 (default 10).
    """

    def __init__(self, method: str = "percentile", **kwargs: Any):
        self.method = method
        self.params = kwargs
        self.is_fitted = False
        self.calibration_params: dict = {}
        self.iso_reg = None

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray) -> "ProbabilityCalibrator":
        val_probs = np.asarray(val_probs, dtype=np.float64).ravel()
        val_labels = np.asarray(val_labels, dtype=np.int64).ravel()

        if self.method == "percentile":
            self._fit_percentile(val_probs, val_labels)
        elif self.method == "beta":
            self._fit_beta(val_probs, val_labels)
        elif self.method == "isotonic":
            self._fit_isotonic(val_probs, val_labels)
        elif self.method == "temperature":
            self._fit_temperature(val_probs, val_labels)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted yet")
        probs = np.asarray(probs, dtype=np.float64).ravel()

        if self.method == "percentile":
            return self._transform_percentile(probs)
        if self.method == "beta":
            return self._transform_beta(probs)
        if self.method == "isotonic":
            return self._transform_isotonic(probs)
        if self.method == "temperature":
            return self._transform_temperature(probs)
        raise ValueError(f"Unknown calibration method: {self.method}")

    # ------------------------------------------------------------------ percentile
    def _fit_percentile(self, val_probs, val_labels):
        target_percentile = self.params.get("target_preictal_percentile", 10)
        preictal_probs = val_probs[val_labels == 1]

        if len(preictal_probs) == 0:
            logger.warning("No preictal samples in validation set; using identity")
            self.calibration_params = {"a": 1.0, "b": 0.0}
            return

        # Map the (100 - target_percentile)-th percentile of preictal probs to 0.5.
        target_prob = np.percentile(preictal_probs, 100 - target_percentile)
        high_percentile_value = 100 - max(target_percentile - 40, 1)
        high_prob = np.percentile(preictal_probs, high_percentile_value)

        try:
            target_logit = _logit(np.clip(target_prob, 1e-7, 1 - 1e-7))
            high_logit = _logit(np.clip(high_prob, 1e-7, 1 - 1e-7))
            if abs(high_logit - target_logit) < 1e-6:
                a = 1.0
            else:
                a = 2.197 / (high_logit - target_logit)
            b = -a * target_logit
        except (ValueError, ZeroDivisionError):
            logger.warning("Failed to compute logit transform; using identity")
            a, b = 1.0, 0.0

        self.calibration_params = {"a": a, "b": b, "target_prob": float(target_prob)}
        logger.info("Percentile calibration: a=%.4f b=%.4f target_prob=%.4f", a, b, target_prob)

    def _transform_percentile(self, probs):
        a = self.calibration_params["a"]
        b = self.calibration_params["b"]
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        return _expit(a * _logit(probs_clipped) + b)

    # ------------------------------------------------------------------ beta
    def _fit_beta(self, val_probs, val_labels):
        opt = _require("scipy.optimize", "scipy")
        p = np.clip(val_probs, 1e-7, 1 - 1e-7)

        def loss_fn(params):
            a, b, c = params
            calibrated = _expit(a + b * np.log(p) + c * np.log(1 - p))
            return -np.mean(
                val_labels * np.log(calibrated + 1e-7)
                + (1 - val_labels) * np.log(1 - calibrated + 1e-7)
            )

        result = opt.minimize(loss_fn, [0.0, 1.0, 1.0], method="L-BFGS-B")
        self.calibration_params = {"a": float(result.x[0]), "b": float(result.x[1]), "c": float(result.x[2])}
        logger.info("Beta calibration: a=%.4f b=%.4f c=%.4f", *result.x)

    def _transform_beta(self, probs):
        p = np.clip(probs, 1e-7, 1 - 1e-7)
        a = self.calibration_params["a"]
        b = self.calibration_params["b"]
        c = self.calibration_params["c"]
        return _expit(a + b * np.log(p) + c * np.log(1 - p))

    # ------------------------------------------------------------------ isotonic
    def _fit_isotonic(self, val_probs, val_labels):
        iso = _require("sklearn.isotonic", "scikit-learn")
        self.iso_reg = iso.IsotonicRegression(out_of_bounds="clip")
        self.iso_reg.fit(val_probs, val_labels)
        logger.info("Fitted isotonic regression calibration")

    def _transform_isotonic(self, probs):
        return self.iso_reg.predict(probs)

    # ------------------------------------------------------------------ temperature
    def _fit_temperature(self, val_probs, val_labels):
        opt = _require("scipy.optimize", "scipy")
        p = np.clip(val_probs, 1e-7, 1 - 1e-7)

        def loss_fn(params):
            T, b = params
            if T <= 0:
                return 1e10
            calibrated = _expit((_logit(p) - b) / T)
            return -np.mean(
                val_labels * np.log(calibrated + 1e-7)
                + (1 - val_labels) * np.log(1 - calibrated + 1e-7)
            )

        result = opt.minimize(loss_fn, [1.0, 0.0], method="L-BFGS-B", bounds=[(0.1, 10.0), (-5.0, 5.0)])
        self.calibration_params = {"T": float(result.x[0]), "b": float(result.x[1])}
        logger.info("Temperature calibration: T=%.4f b=%.4f", *result.x)

    def _transform_temperature(self, probs):
        p = np.clip(probs, 1e-7, 1 - 1e-7)
        T = self.calibration_params["T"]
        b = self.calibration_params["b"]
        return _expit((_logit(p) - b) / T)


def calibrate_ensemble(
    test_probs_stack: np.ndarray,
    val_probs_list: List[np.ndarray],
    val_labels_list: List[np.ndarray],
    val_aucs: np.ndarray,
    calibration_method: str = "percentile",
    **calibration_params: Any,
) -> Tuple[np.ndarray, List[ProbabilityCalibrator]]:
    """Calibrate per-fold then AUC-weight-ensemble test probabilities.

    Parameters
    ----------
    test_probs_stack : (n_folds, n_test)
    val_probs_list   : list of (n_val,) validation probabilities per fold
    val_labels_list  : list of (n_val,) validation labels per fold
    val_aucs         : (n_folds,) validation AUCs used as ensemble weights

    Returns
    -------
    final_probs, calibrators
    """
    n_models = len(val_probs_list)
    calibrators: List[ProbabilityCalibrator] = []
    calibrated_test_probs = []

    for i in range(n_models):
        cal = ProbabilityCalibrator(method=calibration_method, **calibration_params)
        cal.fit(val_probs_list[i], val_labels_list[i])
        calibrators.append(cal)
        calibrated_test_probs.append(cal.transform(test_probs_stack[i]))

    calibrated_stack = np.stack(calibrated_test_probs)
    weights = np.asarray(val_aucs, dtype=np.float64)
    if weights.size == 0 or not np.isfinite(weights).any() or weights.sum() <= 0:
        weights = np.ones(n_models) / max(1, n_models)
    else:
        weights = np.clip(weights, 0, None)
        weights = weights / weights.sum()
    final_probs = np.tensordot(weights, calibrated_stack, axes=1)
    logger.info("Calibrated ensemble predictions using method: %s", calibration_method)
    return final_probs, calibrators
