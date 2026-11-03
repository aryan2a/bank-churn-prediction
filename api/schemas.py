"""Request and response shapes. Limits come from churn.config, so the API and CSV checks agree."""
from typing import Literal

from pydantic import BaseModel, Field

from churn.config import NUMERIC_LIMITS as L


class CustomerIn(BaseModel):
    credit_score: int = Field(..., ge=L["credit_score"][0], le=L["credit_score"][1], description="Credit score")
    country: Literal["France", "Germany", "Spain"]
    gender: Literal["Male", "Female"]
    age: int = Field(..., ge=L["age"][0], le=L["age"][1], description="Age in years")
    tenure: int = Field(..., ge=L["tenure"][0], le=L["tenure"][1], description="Years as a customer")
    balance: float = Field(..., ge=L["balance"][0], le=L["balance"][1], description="Account balance")
    products: int = Field(..., ge=L["products"][0], le=L["products"][1], description="Number of bank products held (1-4)")
    credit_card: bool = Field(..., description="Has a credit card")
    active: bool = Field(..., description="Active member")
    salary: float = Field(..., ge=L["salary"][0], le=L["salary"][1], description="Estimated annual salary")

    model_config = {"json_schema_extra": {"examples": [{
        "credit_score": 600, "country": "Germany", "gender": "Male", "age": 45, "tenure": 3,
        "balance": 120000, "products": 1, "credit_card": True, "active": False, "salary": 55000}]}}


class Driver(BaseModel):
    feature: str
    label: str
    value: str
    impact: float = Field(..., description="SHAP push on the score (log-odds for XGBoost); positive raises risk")
    direction: Literal["raises risk", "lowers risk"]


class PredictionOut(BaseModel):
    risk_score: float = Field(..., description="0-1 ranking score. NOT a calibrated probability.")
    flagged: bool = Field(..., description="True if risk_score is at or above the decision threshold")
    threshold: float
    risk_percentile: float = Field(..., description="Share of customers (out-of-fold scores) with a score at or below this one, 0-100")
    risk_band: Literal["Lowest", "Low", "Medium", "High", "Highest"]
    drivers: list[Driver] | None = Field(None, description="Biggest reasons behind the score (SHAP), strongest first")
    model_name: str
    note: str


class BatchIn(BaseModel):
    customers: list[CustomerIn] = Field(..., min_length=1, max_length=5000)


class BatchRow(BaseModel):
    index: int = Field(..., description="Position of the customer in the request")
    risk_score: float
    flagged: bool
    risk_percentile: float
    risk_band: str
    top_driver: str | None = None


class BatchOut(BaseModel):
    n: int
    n_flagged: int
    threshold: float
    results: list[BatchRow]


class ModelInfo(BaseModel):
    model_name: str
    threshold: float
    threshold_rule: str
    test_scores: dict
    test_ci: dict
    best_params: dict
    cv_results: list
    trained_at: str | None
    n_customers: int | None
    churn_rate: float | None
    versions: dict
    band_edges: list[float]
    threshold_percentile: float | None = Field(None, description="Where the call threshold sits when customers are ranked by out-of-fold score, 0-100")
    explanations: bool
    note: str
