---
title: Credit Risk Scorer
emoji: 💳
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 💳 Credit Risk Scorer

An interactive credit-default model that explains **every** decision. Move the sliders and
watch the probability of default, the approve/decline call, and a **per-applicant SHAP
breakdown** update live — plus the business trade-off (approval rate vs. defaulters caught)
at your chosen threshold.

**Live demo:** _deploy to Hugging Face Spaces / Render — URL goes here (see "Deploy")._

---

## Problem → Approach → Result

- **Problem:** A default-risk score is only useful if it is **explainable** and tied to a
  **business threshold** — "decline" with no reason and no cost trade-off is not actionable.
- **Approach:** An XGBoost classifier served by a FastAPI endpoint. Each `/score` call returns
  the probability, an approve/decline decision, and the top **SHAP** contributions (XGBoost's
  native TreeSHAP). A static slider UI renders it all and shows the approval/recall trade-off.
- **Result:** A deployable scoring API where you can see *why* each decision was made and *what
  it costs* to move the threshold — not just an accuracy number.

## Insights

- **AUC is the least interesting number.** 0.7689 is ordinary; the value is the **threshold curve** —
  at 0.30 you catch 87% of defaulters but approve only 35% of applicants; at 0.70 you approve 85% but
  catch 42%. Picking a threshold is a *business* decision, and the app makes that trade-off explicit
  instead of hiding it behind one accuracy figure.
- **With a 22% default base rate, accuracy lies.** A "78%-accurate" model that approves everyone is
  useless — which is why this reports **AUC-PR (0.5355)** and a recall/precision curve, not accuracy.
- **Explainability has to be per-decision.** Native TreeSHAP returns the contributions behind *this*
  applicant's score (summing in log-odds to the model margin) — what an adjudicator or a regulator
  actually asks for, not a global feature-importance chart.

## Model performance (held-out test, copied from `train.py` output)

- **AUC-ROC: 0.7689** · **AUC-PR: 0.5355** on 6,000 held-out applicants (24,000 train).
- Dataset positive (default) rate: **22.1%**; class imbalance handled with `scale_pos_weight`.

### The business trade-off (`metrics.json`, test set)

| Threshold | Approved | Defaulters caught (recall) | Of declines, truly default (precision) |
| --- | --- | --- | --- |
| 0.30 | 35% | 87% | 30% |
| 0.50 | 71% | 61% | 46% |
| 0.70 | 85% | 42% | 60% |

Lower thresholds catch more defaulters but reject more good applicants — the slider in the app
lets you explore this directly.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements-dev.txt
python train.py                 # fetches UCI data, writes models/*
uvicorn main:app --reload       # UI at http://localhost:8000 , docs at /docs
```

The trained `models/` artifacts are committed, so you can skip `train.py` and just serve.

## Test

```bash
python -m pytest
```

21 tests cover model loading, the API contract, a **frozen golden prediction** (pins serving
behavior to the committed model), that SHAP contributions + base equal the model margin, and
that the decision flips with the threshold.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Slider UI |
| GET | `/health` | Liveness + headline AUC |
| GET | `/spec` | Feature list + UI metadata |
| GET | `/metrics` | Held-out metrics + threshold curve |
| POST | `/score` | `{features, threshold}` → probability, decision, SHAP contributions |

```bash
curl -s localhost:8000/score -H "Content-Type: application/json" -d '{
  "features": {"LIMIT_BAL":50000,"AGE":30,"PAY_0":2,"PAY_2":2,"PAY_3":0,
               "BILL_AMT1":48000,"PAY_AMT1":1500,"PAY_AMT2":1200,
               "EDUCATION":2,"MARRIAGE":2,"SEX":2},
  "threshold": 0.5 }'
```

## Docker

```bash
docker build -t credit-risk .
docker run -p 7860:7860 credit-risk   # http://localhost:7860
```

## Deploy (free)

- **Hugging Face Spaces (Docker SDK):** create a Space → SDK *Docker* → push this repo. Add
  `app_port: 7860` to the Space README frontmatter. The image already listens on 7860.
- **Render:** new Web Service → *Docker* → this repo. Render injects `$PORT`, which the
  container respects.

## Project layout

```text
train.py            # offline training — fetch UCI, fit XGBoost, write models/*
model_service.py    # load model + spec, validate input, probability + TreeSHAP contributions
main.py             # FastAPI: /score /health /spec /metrics + static UI
static/             # vanilla-JS slider UI (no framework)
models/             # committed: model.json, feature_spec.json, metrics.json
data/               # UCI cache (gitignored); see data/README.md
tests/              # pytest: model service + API contract + golden prediction
Dockerfile          # lean serving image (no training deps)
```

## Stack

Python · XGBoost · FastAPI · Pydantic · Docker · vanilla JS. SHAP via XGBoost native TreeSHAP.

---
_Educational demo on public data (UCI Default of Credit Card Clients, 2005). Estimates only —
**not lending advice** and not a real credit decision._
