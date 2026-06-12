"""Serving logic for the credit-default scorer.

Loads the trained booster + feature spec and turns a dict of feature values into a
probability of default plus per-feature SHAP contributions (XGBoost's native
TreeSHAP via ``pred_contribs``). Pure Python/NumPy — no web framework here, so it is
unit-testable on its own.

Feature ORDER always comes from feature_spec.json; nothing is hardcoded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb

MODELS = Path(__file__).parent / "models"


class ModelService:
    def __init__(self, models_dir: Path | str = MODELS):
        models_dir = Path(models_dir)
        self.spec = json.loads((models_dir / "feature_spec.json").read_text())
        self.metrics = json.loads((models_dir / "metrics.json").read_text())
        self.feature_names = [f["name"] for f in self.spec["features"]]
        self.booster = xgb.Booster()
        self.booster.load_model(str(models_dir / "model.json"))

    def _matrix(self, features: dict) -> xgb.DMatrix:
        missing = [n for n in self.feature_names if n not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        try:
            row = [[float(features[n]) for n in self.feature_names]]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"All feature values must be numeric: {exc}") from exc
        return xgb.DMatrix(np.asarray(row, dtype=float), feature_names=self.feature_names)

    def score(self, features: dict, top_k: int = 8) -> dict:
        """Return probability of default, log-odds, and signed SHAP contributions.

        Contributions are in log-odds units (they sum with the base value to the
        model's margin); positive pushes risk UP, negative pushes risk DOWN.
        """
        matrix = self._matrix(features)
        margin = float(self.booster.predict(matrix, output_margin=True)[0])
        probability = float(1.0 / (1.0 + np.exp(-margin)))

        contribs = self.booster.predict(matrix, pred_contribs=True)[0]
        base = float(contribs[-1])
        per_feature = [
            {"feature": name, "value": float(features[name]), "contribution": float(c)}
            for name, c in zip(self.feature_names, contribs[:-1])
        ]
        per_feature.sort(key=lambda d: abs(d["contribution"]), reverse=True)

        return {
            "probability": probability,
            "log_odds": margin,
            "base_log_odds": base,
            "contributions": per_feature[:top_k],
        }

    def decide(self, features: dict, threshold: float = 0.5, top_k: int = 8) -> dict:
        """Score plus an approve/decline decision at the given probability threshold."""
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1).")
        result = self.score(features, top_k=top_k)
        result["threshold"] = float(threshold)
        result["decision"] = "decline" if result["probability"] >= threshold else "approve"
        return result
