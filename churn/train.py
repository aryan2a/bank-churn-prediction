"""Train, tune, evaluate and save the churn model. Same recipe as the notebook, as callable steps.

    python -m churn.train                       # full run: compare 3 models, tune the winner, pick threshold
    python -m churn.train --data path/to.csv    # use another file
    python -m churn.train --fast                # tiny run for tests / smoke checks (not for real use)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from churn import config
from churn.data import load_raw
from churn.features import add_features

PARAM_GRIDS = {
    "Random Forest": {"model__n_estimators": [200, 400], "model__max_depth": [8, 12, 16, None],
                      "model__min_samples_leaf": [2, 3, 5, 10]},
    "XGBoost": {"model__n_estimators": [200, 400], "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.03, 0.05, 0.1], "model__subsample": [0.7, 0.9]},
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10, 100]},
}


def make_preprocess() -> ColumnTransformer:
    # drop="if_binary": a 0/1 column becomes ONE column instead of two mirror-image ones
    return ColumnTransformer([
        ("num", StandardScaler(), config.NUM_COLS_MODEL + config.ENGINEERED),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), config.CAT_COLS_MODEL),
    ])


def build_classifier(name: str, scale: float, **params):
    rs = config.RANDOM_STATE
    if name == "Random Forest":
        return RandomForestClassifier(**params, class_weight="balanced", random_state=rs, n_jobs=-1)
    if name == "XGBoost":
        return XGBClassifier(**params, scale_pos_weight=scale, eval_metric="logloss", random_state=rs, n_jobs=-1)
    if name == "Logistic Regression":
        return LogisticRegression(**params, class_weight="balanced", max_iter=1000, random_state=rs)
    raise ValueError(f"Unknown model: {name}")


def split(X: pd.DataFrame, y: pd.Series):
    """70% train, 15% validation (threshold choice), 15% test (touched once). Same seeds as the notebook."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=config.RANDOM_STATE, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=config.RANDOM_STATE, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def compare_models(X_train_fe, y_train, cv) -> pd.DataFrame:
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    candidates = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", max_depth=8,
                                                random_state=config.RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, scale_pos_weight=scale,
                                 eval_metric="logloss", random_state=config.RANDOM_STATE, n_jobs=-1),
    }
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1", "recall": "recall", "precision": "precision"}
    rows = []
    for name, clf in candidates.items():
        s = cross_validate(Pipeline([("prep", make_preprocess()), ("model", clf)]), X_train_fe, y_train, cv=cv, scoring=scoring)
        rows.append({"model": name, "roc_auc": s["test_roc_auc"].mean(), "roc_auc_std": s["test_roc_auc"].std(),
                     "pr_auc": s["test_pr_auc"].mean(), "f1": s["test_f1"].mean(),
                     "recall": s["test_recall"].mean(), "precision": s["test_precision"].mean()})
    return pd.DataFrame(rows).sort_values(config.SELECT_METRIC, ascending=False).reset_index(drop=True)


def choose_threshold(y_val, proba_val, rule: str = "f1") -> tuple[float, pd.DataFrame]:
    rows = []
    for t in config.THRESHOLD_GRID:
        pred = (proba_val >= t).astype(int)
        flagged = int(pred.sum())
        tp = int(((pred == 1) & (np.asarray(y_val) == 1)).sum())
        rows.append({"threshold": t,
                     "precision": precision_score(y_val, pred, zero_division=0),
                     "recall": recall_score(y_val, pred, zero_division=0),
                     "f1": f1_score(y_val, pred, zero_division=0),
                     "net_benefit": tp * config.SUCCESS_RATE * config.VALUE_PER_SAVE - flagged * config.OFFER_COST})
    table = pd.DataFrame(rows)
    if rule == "f1":
        best = float(table.sort_values("f1", ascending=False).iloc[0]["threshold"])
    elif rule == "cost":
        best = float(table.loc[table["net_benefit"].idxmax(), "threshold"])
    else:
        raise ValueError("threshold rule must be 'f1' or 'cost'")
    return best, table


def evaluate(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {"accuracy": round(accuracy_score(y_true, pred), 4), "precision": round(precision_score(y_true, pred), 4),
            "recall": round(recall_score(y_true, pred), 4), "f1": round(f1_score(y_true, pred), 4),
            "roc_auc": round(roc_auc_score(y_true, proba), 4), "pr_auc": round(average_precision_score(y_true, proba), 4)}


def bootstrap_ci(y_true, proba, threshold: float, n_boot: int = 1000) -> dict:
    """95% ranges: re-draw the test set with replacement n_boot times (notebook section 15)."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    y_arr, n = np.asarray(y_true), len(y_true)
    boot = {"recall": [], "precision": [], "f1": [], "roc_auc": []}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yt, ps = y_arr[i], proba[i]
        if yt.sum() in (0, n):
            continue
        pr = (ps >= threshold).astype(int)
        boot["recall"].append(recall_score(yt, pr))
        boot["precision"].append(precision_score(yt, pr, zero_division=0))
        boot["f1"].append(f1_score(yt, pr))
        boot["roc_auc"].append(roc_auc_score(yt, ps))
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in boot.items()}


def train(df: pd.DataFrame, threshold_rule: str = "f1", fast: bool = False, log=print) -> dict:
    """Run the whole recipe on a frame with the raw features and the churn label. Returns the bundle to save."""
    import sklearn
    X, y = df[config.RAW_FEATURES], df[config.TARGET_COL]
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)
    X_train_fe, X_val_fe, X_test_fe = (add_features(a) for a in (X_train, X_val, X_test))
    cv = StratifiedKFold(5 if not fast else 3, shuffle=True, random_state=config.RANDOM_STATE)

    if fast:
        best_name, cv_results = "XGBoost", []
        scale = (y_train == 0).sum() / (y_train == 1).sum()
        tuned = Pipeline([("prep", make_preprocess()), ("model", build_classifier("XGBoost", scale, n_estimators=40, max_depth=3, learning_rate=0.1))])
        tuned.fit(X_train_fe, y_train)
        best_params = {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.1}
    else:
        log("1/5  comparing Logistic Regression, Random Forest and XGBoost with 5-fold cross-validation ...")
        results = compare_models(X_train_fe, y_train, cv)
        cv_results, best_name = results.round(4).to_dict("records"), results.loc[0, "model"]
        log(f"     best by {config.SELECT_METRIC}: {best_name}")
        log("2/5  tuning the winner with grid search ...")
        scale = (y_train == 0).sum() / (y_train == 1).sum()
        base = Pipeline([("prep", make_preprocess()), ("model", build_classifier(best_name, scale) if best_name != "Random Forest" else RandomForestClassifier(class_weight="balanced", random_state=config.RANDOM_STATE, n_jobs=-1))])
        grid = GridSearchCV(base, PARAM_GRIDS[best_name], scoring=config.SELECT_METRIC, cv=cv, n_jobs=-1)
        grid.fit(X_train_fe, y_train)
        tuned, best_params = grid.best_estimator_, {k.replace("model__", ""): v for k, v in grid.best_params_.items()}
        log(f"     best params: {best_params}  (CV {config.SELECT_METRIC} {grid.best_score_:.4f})")

    log("3/5  choosing the decision threshold on the validation set ...")
    threshold, _ = choose_threshold(y_val, tuned.predict_proba(X_val_fe)[:, 1], threshold_rule)
    log(f"     threshold ({threshold_rule}): {threshold}")

    log("4/5  refitting on train+validation, evaluating once on the test set ...")
    X_final, y_final = pd.concat([X_train_fe, X_val_fe]), pd.concat([y_train, y_val])
    scale_final = (y_final == 0).sum() / (y_final == 1).sum()
    final_model = Pipeline([("prep", make_preprocess()), ("model", build_classifier(best_name, scale_final, **best_params))])
    final_model.fit(X_final, y_final)
    proba_test = final_model.predict_proba(X_test_fe)[:, 1]
    test_scores = evaluate(y_test, proba_test, threshold)
    test_ci = bootstrap_ci(y_test, proba_test, threshold, n_boot=200 if fast else 1000)
    log("     test:", test_scores)

    log("5/5  out-of-fold scores for every customer (for percentiles and risk bands) ...")
    X_all_fe = add_features(X)
    oof = cross_val_predict(clone(final_model), X_all_fe, y, cv=cv, method="predict_proba")[:, 1]
    edges = np.quantile(oof, [0.2, 0.4, 0.6, 0.8])

    import xgboost
    return {
        "model": final_model, "threshold": threshold, "model_name": best_name, "test_scores": test_scores,   # same keys as the notebook's file
        "threshold_rule": threshold_rule, "test_ci": test_ci, "best_params": best_params, "cv_results": cv_results,
        "oof_scores": np.sort(oof).astype("float32"), "band_edges": edges,
        "n_customers": int(len(df)), "churn_rate": round(float(y.mean()), 4),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }


def save(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    card = {k: v for k, v in bundle.items() if k not in ("model", "oof_scores", "band_edges")}
    card["band_edges"] = [round(float(e), 4) for e in bundle["band_edges"]]
    card["score_note"] = config.SCORE_NOTE
    path.with_name("model_card.json").write_text(json.dumps(card, indent=2, default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=config.DATA_PATH, help="training CSV (default: data/churn_raw.csv)")
    ap.add_argument("--out", type=Path, default=config.MODEL_PATH, help="where to save the model (default: models/churn_model.joblib)")
    ap.add_argument("--threshold-rule", choices=["f1", "cost"], default="f1")
    ap.add_argument("--fast", action="store_true", help="tiny run for tests; do not use for the real model")
    args = ap.parse_args(argv)
    try:
        df = load_raw(args.data)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Loaded {len(df):,} customers, churn rate {df[config.TARGET_COL].mean():.3f}")
    bundle = train(df, args.threshold_rule, args.fast)
    save(bundle, args.out)
    print(f"Saved {args.out}  (+ model_card.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
