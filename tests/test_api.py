import pytest
from fastapi.testclient import TestClient

from api.main import app
from churn import config
from tests.helpers import EXAMPLE


@pytest.fixture()
def client(use_model):
    with TestClient(app) as c:
        yield c


def test_health_and_model_info(client):
    assert client.get("/health").json()["model_loaded"] is True
    info = client.get("/model-info").json()
    assert info["model_name"] == "XGBoost" and 0 < info["threshold"] < 1 and "recall" in info["test_scores"]
    assert 0 < info["threshold_percentile"] < 100


def test_predict_one(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["risk_score"] <= 1 and body["drivers"] and body["note"]
    assert "X-Process-Time-ms" in r.headers


@pytest.mark.parametrize("change", [{"country": "Italy"}, {"age": 12}, {"balance": -1}, {"products": 7}, {"credit_score": 99}, {"gender": "X"}])
def test_invalid_input_is_rejected(client, change):
    assert client.post("/predict", json={**EXAMPLE, **change}).status_code == 422


def test_missing_field_is_rejected(client):
    bad = {k: v for k, v in EXAMPLE.items() if k != "age"}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch(client):
    safe = dict(EXAMPLE, age=28, active=True, country="France", products=2)
    r = client.post("/predict/batch?include_drivers=true", json={"customers": [EXAMPLE, safe]})
    body = r.json()
    assert r.status_code == 200 and body["n"] == 2 and [x["index"] for x in body["results"]] == [0, 1]
    assert body["n_flagged"] == sum(x["flagged"] for x in body["results"]) and body["results"][0]["top_driver"]


def test_batch_size_limits(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
    assert client.post("/predict/batch", json={"customers": [EXAMPLE] * 5001}).status_code == 422


def test_service_says_so_when_model_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "nope.joblib")
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "model_missing", "model_loaded": False, "version": c.get("/health").json()["version"]}
        assert c.post("/predict", json=EXAMPLE).status_code == 503
