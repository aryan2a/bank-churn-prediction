import pandas as pd
import pytest

from churn.data import normalise_columns, validate_features
from churn.features import add_features
from tests.helpers import make_customers


def test_zero_balance_flag_and_age_groups():
    df = pd.DataFrame({"balance": [0.0, 50.0], "age": [30, 31]})
    out = add_features(df)
    assert out["zero_balance"].tolist() == [1, 0]
    assert out["age_group"].tolist() == [0, 1]            # bins are (0,30], (30,40], ...


def test_age_group_covers_the_whole_allowed_range():
    out = add_features(pd.DataFrame({"balance": [1.0, 1.0], "age": [18, 100]}))
    assert out["age_group"].tolist() == [0, 4]


def test_both_column_spellings_are_accepted():
    kaggle = pd.DataFrame({"CustomerId": [1], "Surname": ["x"], "CreditScore": [600], "Geography": ["France"], "Gender": ["Male"],
                           "Age": [40], "Tenure": [2], "Balance": [0.0], "NumOfProducts": [1], "HasCrCard": [1],
                           "IsActiveMember": [1], "EstimatedSalary": [50000.0]})
    lower = pd.DataFrame({"customer_id": [1], "credit_score": [600], "country": ["France"], "gender": ["Male"], "age": [40],
                          "tenure": [2], "balance": [0.0], "products_number": [1], "credit_card": [1], "active_member": [1],
                          "estimated_salary": [50000.0]})
    a, b = validate_features(kaggle), validate_features(lower)
    assert list(a.columns) == list(b.columns) and a.equals(b)


def test_clean_data_passes_and_bad_data_is_explained():
    df = make_customers(50, with_label=False)
    assert len(validate_features(df)) == 50
    bad = df.copy()
    bad.loc[3, "country"] = "Italy"
    bad.loc[5, "age"] = 12
    with pytest.raises(ValueError) as e:
        validate_features(bad)
    assert "country" in str(e.value) and "age" in str(e.value)


def test_missing_column_is_reported():
    with pytest.raises(ValueError, match="Missing column"):
        validate_features(make_customers(5, with_label=False).drop(columns=["balance"]))
