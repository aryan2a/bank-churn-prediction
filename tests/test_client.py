"""The client the Streamlit app uses, in in-process mode (no API_URL)."""
import pandas as pd
import pytest

from app.client import ChurnClient, ClientError
from tests.helpers import make_customers


@pytest.fixture()
def client(use_model, monkeypatch):
    monkeypatch.delenv("API_URL", raising=False)
    return ChurnClient()


def test_batch_returns_inputs_plus_scores(client):
    frame = make_customers(40, seed=3).drop(columns=["churn"], errors="ignore")
    out = client.predict_batch(frame)
    assert len(out) == 40
    for col in ["risk_score", "flagged", "risk_percentile", "risk_band", "top_driver", "age", "country"]:
        assert col in out.columns
    assert out["risk_score"].between(0, 1).all()


def test_batch_accepts_original_dataset_column_names(client):
    frame = pd.DataFrame([{"CustomerId": 77, "CreditScore": 600, "Geography": "Germany", "Gender": "Male", "Age": 45, "Tenure": 3,
                           "Balance": 120000, "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 0, "EstimatedSalary": 55000}])
    out = client.predict_batch(frame)
    assert out.loc[0, "customer_id"] == 77 and 0 <= out.loc[0, "risk_score"] <= 1


def test_batch_rejects_bad_rows_with_a_readable_error(client):
    bad = pd.DataFrame([{"credit_score": 50, "country": "Mars", "gender": "Male", "age": 45, "tenure": 3,
                         "balance": 1, "products": 1, "credit_card": 1, "active": 0, "salary": 1}])
    with pytest.raises(ClientError):
        client.predict_batch(bad)


def test_single_prediction_matches_batch(client):
    row = dict(credit_score=600, country="Germany", gender="Male", age=45, tenure=3, balance=120000, products=1,
               credit_card=True, active=False, salary=55000)
    one = client.predict(row)
    batch = client.predict_batch(pd.DataFrame([row]))
    assert one["risk_score"] == pytest.approx(batch.loc[0, "risk_score"])
