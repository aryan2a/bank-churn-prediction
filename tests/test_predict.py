import numpy as np

from churn.predict import ChurnPredictor
from tests.helpers import EXAMPLE, make_customers


def test_scores_are_valid_and_consistent(use_model):
    p = ChurnPredictor.load()
    scored = p.score_frame(make_customers(200, seed=5, with_label=False))
    assert len(scored) == 200
    assert scored["risk_score"].between(0, 1).all()
    assert (scored["flagged"] == (scored["risk_score"] >= p.threshold)).all()
    assert scored["risk_percentile"].between(0, 100).all()
    assert set(scored["risk_band"]) <= {"Lowest", "Low", "Medium", "High", "Highest"}


def test_higher_score_never_has_lower_percentile_or_band(use_model):
    p = ChurnPredictor.load()
    scored = p.score_frame(make_customers(300, seed=6, with_label=False)).sort_values("risk_score")
    assert scored["risk_percentile"].is_monotonic_increasing
    order = {b: i for i, b in enumerate(["Lowest", "Low", "Medium", "High", "Highest"])}
    assert scored["risk_band"].map(order).is_monotonic_increasing


def test_predict_one_returns_sorted_drivers(use_model):
    out = ChurnPredictor.load().predict_one(EXAMPLE)
    assert 0 <= out["risk_score"] <= 1 and isinstance(out["flagged"], bool)
    impacts = [abs(d["impact"]) for d in out["drivers"]]
    assert impacts == sorted(impacts, reverse=True) and len(impacts) == 5
    assert {d["feature"] for d in out["drivers"]} <= {"age", "country", "active", "products", "gender", "balance", "salary", "credit_score", "tenure", "credit_card"}


def test_risky_profile_scores_above_safe_profile(use_model):
    p = ChurnPredictor.load()
    risky = dict(EXAMPLE, age=58, active=False, country="Germany", products=3)
    safe = dict(EXAMPLE, age=28, active=True, country="France", products=2)
    assert p.predict_one(risky)["risk_score"] > p.predict_one(safe)["risk_score"]


def test_batch_drivers_column(use_model):
    scored = ChurnPredictor.load().score_frame(make_customers(40, seed=7, with_label=False), drivers=True)
    assert "top_driver" in scored.columns and scored["top_driver"].notna().all()
