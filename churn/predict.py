"""Load the saved model bundle and score customers. Used by the API and (optionally) directly by the app."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churn import config
from churn.data import validate_features
from churn.explain import Explainer, drivers_for_row
from churn.features import add_features

log = logging.getLogger("churn.predict")


class ModelNotFound(RuntimeError):
    pass


class ChurnPredictor:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.model = bundle["model"]
        self.threshold = float(bundle["threshold"])
        self.model_name = bundle.get("model_name", "model")
        self._oof_sorted = np.sort(np.asarray(bundle["oof_scores"], dtype=float))
        self._edges = np.asarray(bundle["band_edges"], dtype=float)
        self.explainer = Explainer(self.model)
        self._check_versions()

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path | None = None) -> "ChurnPredictor":
        path = Path(path or config.MODEL_PATH)
        if not path.exists():
            raise ModelNotFound(f"No model file at {path}. Train one first:  python -m churn.train")
        return cls(joblib.load(path))

    def _check_versions(self) -> None:
        import sklearn, xgboost
        saved = self.bundle.get("versions", {})
        now = {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}
        for lib, v in now.items():
            if saved.get(lib) and saved[lib] != v:
                log.warning("Model was trained with %s %s but %s is installed; retrain if you see errors.", lib, saved[lib], v)

    # ------------------------------------------------------------------ scoring
    def _band(self, scores: np.ndarray) -> np.ndarray:
        return np.array(config.RISK_BANDS)[np.digitize(scores, self._edges)]

    def score_frame(self, customers: pd.DataFrame, drivers: bool = False) -> pd.DataFrame:
        """Score a frame of customers (raw feature columns). Returns one row per customer, same order."""
        clean = validate_features(customers)
        feats = add_features(clean[config.RAW_FEATURES])
        # Everything below uses the score as shown (4 decimals), so the flag, percentile and band always
        # agree with the number the user sees (and with the dashboard, which also compares rounded scores)
        scores = np.round(self.model.predict_proba(feats)[:, 1].astype(float), 4)
        out = pd.DataFrame({
            "risk_score": scores,
            "flagged": scores >= self.threshold,
            "risk_percentile": np.round(100 * np.searchsorted(self._oof_sorted, scores, side="right") / len(self._oof_sorted), 1),
            "risk_band": self._band(scores),
        }, index=clean.index)
        if "customer_id" in clean.columns:
            out.insert(0, "customer_id", clean["customer_id"])
        if drivers and self.explainer.supported:
            out["top_driver"] = self.explainer.top_driver(self.explainer.grouped_shap(feats)).map(
                lambda f: config.FEATURE_LABELS.get(f, "No risk driver")).values
        return out

    def predict_one(self, customer: dict, top_n: int = 5) -> dict:
        row = pd.DataFrame([customer])
        scored = self.score_frame(row).iloc[0]
        result = {
            "risk_score": float(scored["risk_score"]),
            "flagged": bool(scored["flagged"]),
            "threshold": self.threshold,
            "risk_percentile": float(scored["risk_percentile"]),
            "risk_band": str(scored["risk_band"]),
            "model_name": self.model_name,
            "note": config.SCORE_NOTE,
            "drivers": None,
        }
        if self.explainer.supported:
            clean = validate_features(row)
            feats = add_features(clean[config.RAW_FEATURES])
            result["drivers"] = drivers_for_row(self.explainer.grouped_shap(feats).iloc[0], clean.iloc[0], top_n)
        return result

    def info(self) -> dict:
        b = self.bundle
        return {
            "model_name": self.model_name,
            "threshold": self.threshold,
            "threshold_rule": b.get("threshold_rule", "f1"),
            "test_scores": b.get("test_scores", {}),
            "test_ci": b.get("test_ci", {}),
            "best_params": b.get("best_params", {}),
            "cv_results": b.get("cv_results", []),
            "trained_at": b.get("trained_at"),
            "n_customers": b.get("n_customers"),
            "churn_rate": b.get("churn_rate"),
            "versions": b.get("versions", {}),
            "band_edges": [round(float(e), 4) for e in self._edges],
            # where the call threshold sits when customers are ranked by out-of-fold score (0-100)
            "threshold_percentile": round(100 * float(np.searchsorted(self._oof_sorted, self.threshold, side="left")) / len(self._oof_sorted), 1),
            "explanations": self.explainer.supported,
            "note": config.SCORE_NOTE,
        }


@lru_cache(maxsize=4)
def _load_cached(path: str) -> ChurnPredictor:
    return ChurnPredictor.load(path)


def get_predictor(path: str | None = None) -> ChurnPredictor:
    """Load the model once per file path (the path is resolved first, so changing MODEL_PATH is respected)."""
    return _load_cached(str(path or config.MODEL_PATH))


get_predictor.cache_clear = _load_cached.cache_clear   # keep the familiar helper for tests/reloads
