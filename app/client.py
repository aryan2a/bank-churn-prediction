"""How the Streamlit app gets scores: from the FastAPI service if API_URL is set, otherwise straight from the model file."""
from __future__ import annotations

import os

import pandas as pd
import requests

from churn.data import validate_features


class ClientError(RuntimeError):
    """A problem the user can act on (API down, model missing, bad data)."""


class ChurnClient:
    def __init__(self, api_url: str | None = None):
        self.api_url = (api_url if api_url is not None else os.getenv("API_URL", "")).rstrip("/")

    @property
    def mode(self) -> str:
        return f"FastAPI service at {self.api_url}" if self.api_url else "in-process model (no API_URL set)"

    # ---------------------------------------------------------------- helpers
    def _get(self, path: str):
        try:
            r = requests.get(f"{self.api_url}{path}", timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.ConnectionError as e:
            raise ClientError(f"Cannot reach the API at {self.api_url}. Is it running? (uvicorn api.main:app)") from e
        except requests.HTTPError as e:
            raise ClientError(f"API error: {e.response.status_code} {e.response.text[:300]}") from e

    def _post(self, path: str, payload: dict, timeout: int = 120):
        try:
            r = requests.post(f"{self.api_url}{path}", json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.ConnectionError as e:
            raise ClientError(f"Cannot reach the API at {self.api_url}. Is it running? (uvicorn api.main:app)") from e
        except requests.HTTPError as e:
            raise ClientError(f"API error: {e.response.status_code} {e.response.text[:300]}") from e

    @staticmethod
    def _predictor():
        from churn.predict import ModelNotFound, get_predictor
        try:
            return get_predictor()
        except ModelNotFound as e:
            raise ClientError(str(e)) from e

    # ---------------------------------------------------------------- public
    def health(self) -> dict:
        if self.api_url:
            try:
                return self._get("/health")
            except ClientError as e:                      # unreachable, or something else is answering on that port
                return {"status": "unreachable", "model_loaded": False, "error": str(e)}
        try:
            self._predictor()
            return {"status": "ok", "model_loaded": True}
        except ClientError:
            return {"status": "model_missing", "model_loaded": False}

    def model_info(self) -> dict:
        return self._get("/model-info") if self.api_url else self._predictor().info()

    def predict(self, customer: dict) -> dict:
        if self.api_url:
            return self._post("/predict", customer)
        return self._predictor().predict_one(customer)

    def predict_batch(self, frame: pd.DataFrame, chunk: int = 1000, progress=None) -> pd.DataFrame:
        """Score a frame of customers; returns the customers' inputs plus the score columns."""
        try:
            clean = validate_features(frame)
        except ValueError as e:
            raise ClientError(str(e)) from e
        clean = clean.reset_index(drop=True)
        if not self.api_url:
            scored = self._predictor().score_frame(clean, drivers=True).drop(columns=["customer_id"], errors="ignore")
        else:
            pieces = []
            for start in range(0, len(clean), chunk):
                part = clean.iloc[start:start + chunk]
                records = part.drop(columns=["customer_id"], errors="ignore").to_dict("records")
                for rec in records:
                    rec["credit_card"], rec["active"] = bool(rec["credit_card"]), bool(rec["active"])
                out = self._post("/predict/batch?include_drivers=true", {"customers": records})
                pieces.append(pd.DataFrame(out["results"]).drop(columns=["index"]))
                if progress:
                    progress(min(start + chunk, len(clean)) / len(clean))
            scored = pd.concat(pieces, ignore_index=True)
        return pd.concat([clean, scored.reset_index(drop=True)], axis=1)
