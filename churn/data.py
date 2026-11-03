"""Loading and checking customer data. Accepts both common spellings of the column names."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn.config import COUNTRIES, GENDERS, NUMERIC_LIMITS, RAW_FEATURES, TARGET_COL

# Every spelling we accept  ->  the lowercase names used everywhere in this project
COLUMN_MAP = {
    "CreditScore": "credit_score", "credit_score": "credit_score",
    "Geography": "country", "country": "country",
    "Gender": "gender", "gender": "gender",
    "Age": "age", "age": "age",
    "Tenure": "tenure", "tenure": "tenure",
    "Balance": "balance", "balance": "balance",
    "NumOfProducts": "products", "products_number": "products", "products": "products",
    "HasCrCard": "credit_card", "credit_card": "credit_card",
    "IsActiveMember": "active", "active_member": "active", "active": "active",
    "EstimatedSalary": "salary", "estimated_salary": "salary", "salary": "salary",
    "Exited": "churn", "churn": "churn",
    "CustomerId": "customer_id", "customer_id": "customer_id",
}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known columns to the project's names and drop the rest (RowNumber, Surname, ...)."""
    keep = {c: COLUMN_MAP[c] for c in df.columns if c in COLUMN_MAP}
    return df[list(keep)].rename(columns=keep)


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Check a frame of customers and return a clean copy with the raw feature columns (plus customer_id).

    Raises ValueError with a readable message instead of failing somewhere deep inside the model.
    """
    df = normalise_columns(df)
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {missing}. Expected: {RAW_FEATURES}")

    out = df.copy()
    problems = []
    for col, (lo, hi) in NUMERIC_LIMITS.items():
        out[col] = pd.to_numeric(out[col], errors="coerce")
        bad = out[col].isna() | (out[col] < lo) | (out[col] > hi)
        if bad.any():
            problems.append(f"{col}: {int(bad.sum())} value(s) missing or outside {lo} to {hi} (first bad row: {int(bad.idxmax())})")
    for col, allowed in (("country", COUNTRIES), ("gender", GENDERS)):
        bad = ~out[col].isin(allowed)
        if bad.any():
            problems.append(f"{col}: {int(bad.sum())} value(s) not in {allowed} (first bad row: {int(bad.idxmax())})")
    for col in ("credit_card", "active"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
        bad = ~out[col].isin([0, 1])
        if bad.any():
            problems.append(f"{col}: {int(bad.sum())} value(s) not 0 or 1 (first bad row: {int(bad.idxmax())})")
    if problems:
        raise ValueError("Invalid data: " + "; ".join(problems))

    for col in ("credit_score", "age", "tenure", "products", "credit_card", "active"):
        out[col] = out[col].astype(int)
    cols = (["customer_id"] if "customer_id" in out.columns else []) + RAW_FEATURES
    return out[cols]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the training CSV; return the validated features plus the churn label."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found at {path}. Put churn_raw.csv in the data/ folder "
                                f"or pass --data <path>.")
    raw = normalise_columns(pd.read_csv(path))
    if TARGET_COL not in raw.columns:
        raise ValueError("The training file needs a churn label column (named 'churn' or 'Exited').")
    feats = validate_features(raw)
    feats[TARGET_COL] = raw[TARGET_COL].astype(int).values
    return feats.reset_index(drop=True)
