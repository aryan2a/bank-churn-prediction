"""Synthetic customers for the tests, so they run anywhere (CI included) without the real dataset."""
import numpy as np
import pandas as pd


def make_customers(n: int = 800, seed: int = 0, with_label: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "credit_score": rng.integers(350, 851, n),
        "country": rng.choice(["France", "Germany", "Spain"], n, p=[0.5, 0.25, 0.25]),
        "gender": rng.choice(["Male", "Female"], n),
        "age": np.clip(rng.normal(39, 10, n).round(), 18, 92).astype(int),
        "tenure": rng.integers(0, 11, n),
        "balance": np.where(rng.random(n) < 0.35, 0.0, rng.normal(120000, 30000, n).clip(0)),
        "products": rng.choice([1, 2, 3, 4], n, p=[0.5, 0.45, 0.04, 0.01]),
        "credit_card": rng.integers(0, 2, n),
        "active": rng.integers(0, 2, n),
        "salary": rng.uniform(10000, 200000, n),
    })
    if with_label:
        logit = -2.6 + 0.07 * (df["age"] - 38) + 0.9 * (df["country"] == "Germany") - 0.9 * df["active"] + 0.6 * (df["products"] >= 3)
        df["churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


EXAMPLE = {"credit_score": 600, "country": "Germany", "gender": "Male", "age": 45, "tenure": 3,
           "balance": 120000, "products": 1, "credit_card": True, "active": False, "salary": 55000}
