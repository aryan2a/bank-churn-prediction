import pytest

from churn import config, train
from tests.helpers import make_customers


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    """Train a tiny model once per test session and save it where the code under test can find it."""
    path = tmp_path_factory.mktemp("model") / "churn_model.joblib"
    bundle = train.train(make_customers(900, seed=1), fast=True, log=lambda *a, **k: None)
    train.save(bundle, path)
    return path


@pytest.fixture()
def use_model(model_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", model_path)
    return model_path
