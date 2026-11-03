"""Per-customer explanations with SHAP, summed back to the ORIGINAL features (as in notebook section 17)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from churn.config import CAT_COLS_MODEL, FEATURE_LABELS

GROUP_OF = {"age_group": "age", "zero_balance": "balance"}   # engineered columns belong to the feature they came from


def parent_feature(name: str) -> str:
    for c in CAT_COLS_MODEL:                                  # one-hot columns look like "country_Germany"
        if name == c or name.startswith(c + "_"):
            return c
    return GROUP_OF.get(name, name)


def _dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


class Explainer:
    """Works for tree models (XGBoost / Random Forest). For other models `supported` is False."""

    def __init__(self, model):
        self.prep = model.named_steps["prep"]
        self.clf = model.named_steps["model"]
        self.supported = hasattr(self.clf, "feature_importances_")
        if self.supported:
            import shap
            self._explainer = shap.TreeExplainer(self.clf)
            self._names = [n.split("__", 1)[-1] for n in self.prep.get_feature_names_out()]

    def grouped_shap(self, frame_fe: pd.DataFrame) -> pd.DataFrame:
        """Rows = customers, columns = original features, values = push on the score (log-odds for XGBoost)."""
        if not self.supported:
            raise RuntimeError("Explanations are only available for tree models.")
        Xd = _dense(self.prep.transform(frame_fe))
        sv = self._explainer.shap_values(Xd, check_additivity=False)
        if isinstance(sv, list):          # older shap: one array per class
            sv = sv[1]
        elif np.ndim(sv) == 3:            # newer shap: (rows, features, classes)
            sv = sv[:, :, 1]
        return pd.DataFrame(np.asarray(sv), columns=self._names, index=frame_fe.index).T.groupby(parent_feature).sum().T

    @staticmethod
    def top_driver(grouped: pd.DataFrame) -> pd.Series:
        """Feature with the biggest push TOWARDS leaving, per customer (same rule as the dashboard's top_driver)."""
        best = grouped.idxmax(axis=1)
        return best.where(grouped.max(axis=1) > 0, "no_risk_driver")


def describe_value(feature: str, value) -> str:
    if feature in ("credit_card", "active"):
        return "Yes" if int(value) == 1 else "No"
    if feature in ("balance", "salary"):
        return f"{float(value):,.0f}"
    return str(value)


def drivers_for_row(grouped_row: pd.Series, raw_row: pd.Series, top_n: int = 5) -> list[dict]:
    order = grouped_row.abs().sort_values(ascending=False).index[:top_n]
    return [{
        "feature": f,
        "label": FEATURE_LABELS.get(f, f),
        "value": describe_value(f, raw_row[f]),
        "impact": round(float(grouped_row[f]), 4),
        "direction": "raises risk" if grouped_row[f] > 0 else "lowers risk",
    } for f in order]
