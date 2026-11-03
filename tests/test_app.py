"""Run the Streamlit script headlessly (in-process mode) and check that every tab renders without errors."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


@pytest.fixture()
def app(use_model, monkeypatch):
    monkeypatch.delenv("API_URL", raising=False)
    return AppTest.from_file(APP, default_timeout=60)


def test_app_renders_and_scores_the_default_customer(app):
    app.run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert "Risk score" in labels and "Risk band" in labels


def test_example_buttons_change_the_result(app):
    app.run()
    first = [m.value for m in app.metric if m.label == "Risk score"][0]
    [b for b in app.button if b.label == "Example: likely to stay"][0].click().run()
    assert not app.exception
    second = [m.value for m in app.metric if m.label == "Risk score"][0]
    assert float(second) < float(first)


def test_missing_model_shows_a_clear_message(tmp_path, monkeypatch):
    from churn import config
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "none.joblib")
    monkeypatch.delenv("API_URL", raising=False)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and any("no trained model" in w.value for w in at.warning)


def test_app_has_the_three_model_tabs(app):
    app.run()
    assert [t.label for t in app.tabs] == ["Score a customer", "Batch scoring", "About the model"]


def test_unreachable_api_shows_a_helpful_message_not_a_traceback(use_model, monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")      # nothing listens on port 9
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Cannot reach the API" in e.value for e in at.error)
