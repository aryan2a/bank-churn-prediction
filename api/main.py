"""FastAPI service: score customers with the saved churn model.

    uvicorn api.main:app --reload          then open  http://127.0.0.1:8000/docs
"""
import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse

from api.schemas import BatchIn, BatchOut, BatchRow, CustomerIn, ModelInfo, PredictionOut
from churn import __version__, config
from churn.predict import ChurnPredictor, ModelNotFound

log = logging.getLogger("churn.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.predictor = ChurnPredictor.load()
        log.info("Loaded %s from %s", app.state.predictor.model_name, config.MODEL_PATH)
    except ModelNotFound as e:                      # start anyway so /health can say what is wrong
        app.state.predictor = None
        log.error("%s", e)
    yield


app = FastAPI(
    title="Bank Churn Risk API",
    version=__version__,
    description=("Scores bank customers for churn risk. **The risk score is a ranking score, not a calibrated "
                 "probability** (the model is trained with class weights). Customers at or above the threshold "
                 "are flagged for a retention call."),
    lifespan=lifespan,
)


@app.middleware("http")
async def add_timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return response


def get_predictor(request: Request) -> ChurnPredictor:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(503, f"Model not loaded. Train one first: python -m churn.train (expected at {config.MODEL_PATH})")
    return predictor


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", tags=["service"])
def health(request: Request):
    loaded = request.app.state.predictor is not None
    return {"status": "ok" if loaded else "model_missing", "model_loaded": loaded, "version": __version__}


@app.get("/model-info", response_model=ModelInfo, tags=["service"])
def model_info(predictor: ChurnPredictor = Depends(get_predictor)):
    return predictor.info()


@app.post("/predict", response_model=PredictionOut, tags=["scoring"])
def predict(customer: CustomerIn, predictor: ChurnPredictor = Depends(get_predictor)):
    """Score one customer and explain the score with the top SHAP drivers."""
    return predictor.predict_one(customer.model_dump())


@app.post("/predict/batch", response_model=BatchOut, tags=["scoring"])
def predict_batch(batch: BatchIn, include_drivers: bool = False, predictor: ChurnPredictor = Depends(get_predictor)):
    """Score up to 5,000 customers in one call. Set include_drivers=true to get each customer's top driver."""
    frame = pd.DataFrame([c.model_dump() for c in batch.customers])
    scored = predictor.score_frame(frame, drivers=include_drivers)
    rows = [BatchRow(index=i, risk_score=r.risk_score, flagged=bool(r.flagged), risk_percentile=r.risk_percentile,
                     risk_band=r.risk_band, top_driver=getattr(r, "top_driver", None))
            for i, r in enumerate(scored.itertuples(index=False))]
    return BatchOut(n=len(rows), n_flagged=int(scored["flagged"].sum()), threshold=predictor.threshold, results=rows)
