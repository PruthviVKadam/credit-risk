# data/

## UCI Default of Credit Card Clients (id 350)

`train.py` fetches this dataset via the `ucimlrepo` package and caches it here as
`credit_default.csv` (gitignored — re-created on demand, not committed).

- 30,000 credit-card holders in Taiwan, 2005; target = default payment next month
  (positive rate ≈ 22%).
- The model uses a **curated, human-readable subset** of 11 features (credit limit, age,
  recent repayment statuses, recent bill/payment amounts, education, marital status, sex).

### Provenance

Public dataset from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
(Yeh & Lien, 2009). Fetched programmatically with
[`ucimlrepo`](https://pypi.org/project/ucimlrepo/) — no manual download or account required.

The trained artifacts in `../models/` (`model.json`, `feature_spec.json`, `metrics.json`)
ARE committed so the API runs without retraining. Regenerate them with `python train.py`.
