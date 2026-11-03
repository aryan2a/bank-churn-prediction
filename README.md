# Bank customer churn: model, API, app and dashboard

Find the bank customers who are most likely to leave, explain why, and give the retention team a call list.

| Part | What it does | Where |
|---|---|---|
| Notebook | EDA, outlier check, model comparison, tuning, evaluation, SHAP, dashboard export | `notebooks/churn_prediction_notebook.ipynb` |
| `churn/` | Training script and the scoring code the API and app share | `python -m churn.train` |
| `api/` | FastAPI service: `/predict`, `/predict/batch`, `/model-info`, `/health` | `uvicorn api.main:app` |
| `app/` | Streamlit app: score one customer, score a CSV, read about the model | `streamlit run app/streamlit_app.py` |
| Power BI | 3-page dashboard: model performance, churn analysis, retention analytics | `dashboard/` (see its README) |
| `tests/` | 35 tests for the data checks, features, scoring, API, client and app | `pytest` |

**Status:** notebook, model training, API, app, tests and dashboard are done. Next, step by step: Docker, GitHub Actions and the AWS EC2 deployment.

## What the model does, and what it does not

- Data: the public 10,000-customer bank churn file, 20.4% of customers left.
- Model: the best of logistic regression, random forest and XGBoost by 5-fold ROC-AUC (XGBoost), tuned with grid search, then refit on train + validation and tested **once** on 1,500 held-out customers.
- Held-out results at the chosen threshold (0.65, picked on the validation set to maximise F1): recall about 0.66, precision about 0.67, ROC-AUC about 0.88. `python -m churn.train` prints your own numbers with 95% ranges, and the app shows them on the *About the model* tab.
- The **risk score is a ranking score, not a probability.** The model is trained with class weights, so a score of 0.8 does not mean an 80% chance. Use it to order a call list.
- The **drivers** are SHAP values. They explain the model's score, not a customer's real reasons for leaving.
- Gender and country are model inputs. Check fairness and the law before any real use.
- The offer cost, value of a saved customer and success rate on dashboard page 3 are **placeholders**, not facts from this data.
- One public dataset, no time dimension: this shows the method, not how it would behave in a live bank.

## The notebook

`notebooks/churn_prediction_notebook.ipynb` is the analysis behind the project: data checks, outlier check, model comparison, tuning, threshold choice, held-out evaluation, SHAP, and (section 19) the export of the CSVs the Power BI dashboard reads. `churn/train.py` is the same recipe as a script, and it reproduces the notebook's numbers.

Run it from the `notebooks/` folder with `data/churn_raw.csv` in place (the paths are relative to that folder). It writes the dashboard CSVs to `dashboard/data/` and a notebook-only model file `notebooks/churn_model_notebook.joblib` (not the one the app uses). A full run takes about two minutes. Run all cells and save before committing, so GitHub shows the outputs.

## Run it on Windows (PowerShell)

```powershell
# 1. one-time setup (Python 3.10+)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. put the training CSV at data\churn_raw.csv  (both column spellings work)

# 3. train (about a minute); saves models\churn_model.joblib and models\model_card.json
python -m churn.train

# 4. start the API (leave this window open)
uvicorn api.main:app --port 8000
#    docs page: http://127.0.0.1:8000/docs

# 5. in a second PowerShell window, from the same folder
.venv\Scripts\Activate.ps1
$env:API_URL = "http://127.0.0.1:8000"
streamlit run app/streamlit_app.py
#    app: http://localhost:8501
```

The app also works on its own: if you skip step 4 and the `$env:API_URL` line, it loads the model file itself. Use the API route when you want to show the full chain (app, API, model).

If you get *running scripts is disabled* when activating the venv, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

Train options: `--data path.csv`, `--out path.joblib`, `--threshold-rule f1|cost` (the `cost` rule uses the placeholder costs in `churn/config.py`).

## API

```powershell
$body = '{"credit_score":600,"country":"Germany","gender":"Male","age":45,"tenure":3,"balance":120000,"products":1,"credit_card":true,"active":false,"salary":55000}'
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body $body
```

```json
{
  "risk_score": 0.883,
  "flagged": true,
  "threshold": 0.65,
  "risk_percentile": 92.6,
  "risk_band": "Highest",
  "drivers": [
    {"feature": "age", "label": "Age", "value": "45", "impact": 0.7178, "direction": "raises risk"},
    {"feature": "country", "label": "Country", "value": "Germany", "impact": 0.5726, "direction": "raises risk"}
  ],
  "model_name": "XGBoost",
  "note": "The risk score ranks customers; it is not a calibrated probability, ..."
}
```

(Example output; your numbers depend on your own training run.)

| Endpoint | Purpose |
|---|---|
| `GET /health` | Is the service up and the model loaded (for health checks) |
| `GET /model-info` | Model name, threshold, held-out scores, training library versions |
| `POST /predict` | One customer in, score + flag + percentile + band + drivers out |
| `POST /predict/batch?include_drivers=true` | Up to 5,000 customers per call |

Bad input (unknown country, age 5, missing field) returns a 422 that names the field. If no model file exists, scoring endpoints return 503 with the instruction to train first.

`risk_percentile` and `risk_band` are measured against out-of-fold scores of the training customers, the same scores the Power BI dashboard uses, so a customer's band means the same thing in both places.

## The Streamlit app

1. **Score a customer**: inputs on the left; flag, score, percentile, band, gauge and the top drivers on the right. Two example buttons fill in a likely leaver and a likely stayer.
2. **Batch scoring**: upload a CSV (template provided, original dataset column names also accepted), see how many are flagged, filter by band, download all results. Customers the model was trained on get optimistic scores, so use new customers to judge behaviour.
3. **About the model**: held-out results with 95% ranges, candidate comparison, how to read the score, limits.

## Power BI dashboard

The dashboard lives in `dashboard/`, separate from the app: the `.pbix` file, the CSVs it reads, and a screenshot of each page. Open `dashboard/README.md` for what each page shows, the key numbers and the caveats (out-of-fold scores, placeholder costs).

## Tests

```powershell
pytest
```

The tests train a tiny model on synthetic customers, so they run anywhere without the real dataset. They cover input validation, feature engineering, score rounding and percentiles, every API endpoint including error cases, the app client, and the Streamlit app (rendered headlessly).

## Folder map

```
notebooks/    the analysis notebook
churn/        config, data checks, features, training, prediction, SHAP explanations
api/          FastAPI app and request/response schemas
app/          Streamlit app, API client, look and feel (ui.py), bundled fonts (static/fonts)
tests/        pytest suite (synthetic data)
models/       churn_model.joblib + model_card.json (created by training, not committed)
data/         churn_raw.csv (not committed)
dashboard/    Power BI report (.pbix), its CSVs, page screenshots
requirements.txt  requirements-api.txt  requirements-app.txt
```

## Next

- Docker: one image for the API and one for the app, run together with Docker Compose.
- GitHub Actions: run the tests on every push and build the images.
- AWS EC2: run the containers on an instance, with the model file copied to it.

## Credits

The app's fonts, Bricolage Grotesque and Instrument Sans, are bundled in `app/static/fonts/` under the SIL Open Font License (licence files alongside).
