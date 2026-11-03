"""One place for paths, column names, input limits and settings (mirrors the notebook's config cell)."""
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Paths can be overridden with environment variables (used by Docker and by the tests)
DATA_PATH = Path(os.getenv("CHURN_DATA_PATH", ROOT / "data" / "churn_raw.csv"))
MODEL_PATH = Path(os.getenv("CHURN_MODEL_PATH", ROOT / "models" / "churn_model.joblib"))

RANDOM_STATE = 42
TARGET_COL = "churn"

# Raw inputs: what a caller (API, app, CSV upload) has to provide for each customer
RAW_FEATURES = ["credit_score", "country", "gender", "age", "tenure",
                "balance", "products", "credit_card", "active", "salary"]
ENGINEERED = ["zero_balance", "age_group"]
NUM_COLS_MODEL = ["credit_score", "age", "tenure", "balance", "products", "salary"]
CAT_COLS_MODEL = ["country", "gender", "credit_card", "active"]

COUNTRIES = ["France", "Germany", "Spain"]
GENDERS = ["Male", "Female"]

# Allowed range of each numeric input: (min, max). The training data sits well inside these.
# The API schema and the CSV validation both read this table, so the two can never disagree.
NUMERIC_LIMITS = {
    "credit_score": (300, 900),
    "age": (18, 100),
    "tenure": (0, 40),
    "balance": (0.0, 1e8),
    "products": (1, 4),        # the model has only seen 1-4 products
    "salary": (0.0, 1e8),
}

# ---- Model selection settings (same as the notebook) ------------------------------------------------
SELECT_METRIC = "roc_auc"
THRESHOLD_GRID = np.round(np.arange(0.10, 0.90, 0.01), 2)

# ---- ASSUMPTIONS (not facts): only used when --threshold-rule cost is chosen -----------------------
OFFER_COST = 1
VALUE_PER_SAVE = 10
SUCCESS_RATE = 0.30

RISK_BANDS = ["Lowest", "Low", "Medium", "High", "Highest"]   # five equal groups of the out-of-fold scores

FEATURE_LABELS = {
    "age": "Age", "products": "Number of products", "active": "Active member", "country": "Country",
    "gender": "Gender", "balance": "Balance", "salary": "Salary", "credit_score": "Credit score",
    "tenure": "Tenure", "credit_card": "Has credit card",
}

SCORE_NOTE = ("The risk score ranks customers; it is not a calibrated probability, because the model is "
              "trained with class weights. Use it to order a call list, not to quote a percentage.")
