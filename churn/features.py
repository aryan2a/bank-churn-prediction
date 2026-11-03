"""Feature engineering: the two columns the notebook adds and validates in section 8."""
import pandas as pd


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["zero_balance"] = (out["balance"] == 0).astype(int)           # marks the spike of customers with no balance
    out["age_group"] = pd.cut(                                        # bins match the observed jump in churn after 40
        out["age"], bins=[0, 30, 40, 50, 60, 100], labels=[0, 1, 2, 3, 4]
    ).astype(int)
    return out
