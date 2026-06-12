# Credit Risk Scorer — CLAUDE.md
XGBoost default-probability model served via FastAPI with SHAP explanations; slider UI in static HTML/JS. Python 3.11+.

## Commands
- Setup: pip install -r requirements-dev.txt
- Train: python train.py (writes models/model.json + models/feature_spec.json + models/metrics.json)
- Serve: uvicorn main:app --reload (UI at /, API docs at /docs)
- Test: python -m pytest (API contract + model loading + a frozen golden-prediction test)
- Container: docker build -t credit-risk . && docker run -p 7860:7860 credit-risk

## Structure
- train.py (offline only — never runs in the API) · model_service.py (scoring + SHAP) · main.py (FastAPI) · static/ (UI)
- models/ committed (model.json + feature_spec.json + metrics.json); raw data is NOT committed (.gitignored) — see data/README.md

## Data & SHAP notes
- Dataset: UCI "Default of Credit Card Clients" (id 350), fetched via ucimlrepo (no Kaggle/account).
- SHAP: XGBoost native TreeSHAP via booster.predict(pred_contribs=True) — no `shap` package, lean image. Contributions are in log-odds and sum with the base value to the margin.

## Rules
- /score returns probability + top-8 SHAP contributions; never return a bare yes/no.
- Train/serve feature order must come from feature_spec.json — no hardcoded column lists.
- UI must label outputs as estimates from a public-data demo, not lending advice.
- Report AUC only from the held-out test split produced by train.py; copy the number, never round up.
- The golden-prediction test pins serving behavior to the committed model.json; if you retrain, regenerate the golden constants deliberately.
