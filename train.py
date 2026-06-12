"""Offline training for the credit-default scorer.

Run:  python train.py

Fetches the UCI "Default of Credit Card Clients" dataset (30,000 Taiwanese credit-card
holders, 2005), trains an XGBoost classifier on a curated, human-readable feature set,
and writes three artifacts the API serves:

    models/model.json          - the trained booster
    models/feature_spec.json   - ordered feature list + UI metadata (ranges, labels)
    models/metrics.json        - held-out AUC + a business threshold curve

This file is NEVER imported by the API. All numbers in the README come from its output.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
DATA = ROOT / "data"

RANDOM_STATE = 42
UCI_ID = 350

# Repayment-status help text shared by the PAY_* (delay) features.
PAY_HELP = "-2/-1/0 = paid on time or no balance; 1-9 = months of payment delay."

# Curated features: UCI code -> friendly name + UI metadata. Order here is the
# canonical model + serving order (also written to feature_spec.json).
FEATURES = [
    {"uci": "X1", "name": "LIMIT_BAL", "label": "Credit limit (NT$)", "step": 1000, "help": ""},
    {"uci": "X5", "name": "AGE", "label": "Age (years)", "step": 1, "help": ""},
    {"uci": "X6", "name": "PAY_0", "label": "Repayment status — last month", "step": 1, "help": PAY_HELP},
    {"uci": "X7", "name": "PAY_2", "label": "Repayment status — 2 months ago", "step": 1, "help": PAY_HELP},
    {"uci": "X8", "name": "PAY_3", "label": "Repayment status — 3 months ago", "step": 1, "help": PAY_HELP},
    {"uci": "X12", "name": "BILL_AMT1", "label": "Most recent bill (NT$)", "step": 1000, "help": ""},
    {"uci": "X18", "name": "PAY_AMT1", "label": "Most recent payment (NT$)", "step": 1000, "help": ""},
    {"uci": "X19", "name": "PAY_AMT2", "label": "Payment — 2 months ago (NT$)", "step": 1000, "help": ""},
    {"uci": "X3", "name": "EDUCATION", "label": "Education (1 grad … 4 other)", "step": 1, "help": ""},
    {"uci": "X4", "name": "MARRIAGE", "label": "Marital status (1 married, 2 single, 3 other)", "step": 1, "help": ""},
    {"uci": "X2", "name": "SEX", "label": "Sex (1 male, 2 female)", "step": 1, "help": ""},
]

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def load_data() -> tuple["object", "object"]:
    """Fetch UCI 350 and return (X with friendly names, y). Caches a CSV under data/."""
    dataset = fetch_ucirepo(id=UCI_ID)
    x_all = dataset.data.features
    y = dataset.data.targets.iloc[:, 0].astype(int)
    rename = {f["uci"]: f["name"] for f in FEATURES}
    x = x_all[[f["uci"] for f in FEATURES]].rename(columns=rename)

    DATA.mkdir(exist_ok=True)
    cache = x.copy()
    cache["DEFAULT"] = y.values
    cache.to_csv(DATA / "credit_default.csv", index=False)
    return x, y


def threshold_curve(y_true: np.ndarray, proba: np.ndarray) -> list[dict]:
    """Business view: at each decision threshold, what do we approve and catch?"""
    y_true = np.asarray(y_true)
    positives = int((y_true == 1).sum())
    rows = []
    for t in THRESHOLDS:
        flagged = proba >= t
        n_flagged = int(flagged.sum())
        true_pos = int((flagged & (y_true == 1)).sum())
        rows.append(
            {
                "threshold": float(t),
                "approval_rate": round(float((~flagged).mean()), 4),
                "recall": round(true_pos / positives, 4) if positives else 0.0,
                "precision": round(true_pos / n_flagged, 4) if n_flagged else 0.0,
            }
        )
    return rows


def main() -> None:
    x, y = load_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_lambda=1.0,
        scale_pos_weight=n_neg / n_pos,   # handle the ~22% positive rate
        eval_metric="auc",
        tree_method="hist",
        n_jobs=4,
        random_state=RANDOM_STATE,
    )
    clf.fit(x_train, y_train)

    proba_test = clf.predict_proba(x_test)[:, 1]
    auc_roc = float(roc_auc_score(y_test, proba_test))
    auc_pr = float(average_precision_score(y_test, proba_test))

    MODELS.mkdir(exist_ok=True)
    clf.get_booster().save_model(str(MODELS / "model.json"))

    spec_features = []
    for f in FEATURES:
        col = x[f["name"]]
        spec_features.append(
            {
                "name": f["name"],
                "label": f["label"],
                "min": float(col.min()),
                "max": float(col.max()),
                "default": float(col.median()),
                "step": f["step"],
                "help": f["help"],
            }
        )
    spec = {
        "features": spec_features,
        "target": "probability of default next month",
        "model": "xgboost binary:logistic",
        "source": "UCI Default of Credit Card Clients (id 350)",
    }
    (MODELS / "feature_spec.json").write_text(json.dumps(spec, indent=2))

    metrics = {
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_features": len(FEATURES),
        "positive_rate": round(float(y.mean()), 4),
        "default_threshold": 0.5,
        "trained_on": date.today().isoformat(),
        "threshold_curve": threshold_curve(y_test.values, proba_test),
    }
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps({k: v for k, v in metrics.items() if k != "threshold_curve"}, indent=2))
    print(f"Wrote model + spec + metrics to {MODELS}/")


if __name__ == "__main__":
    main()
