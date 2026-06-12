"""Tests for ModelService.

The golden values are frozen from the committed model.json, so they pin the exact
serving behavior; if the model is retrained, regenerate them deliberately.
"""

import math

import pytest

from model_service import ModelService

EXPECTED_ORDER = [
    "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3",
    "BILL_AMT1", "PAY_AMT1", "PAY_AMT2", "EDUCATION", "MARRIAGE", "SEX",
]

# A moderately risky applicant (2 months behind on recent payments).
GOLDEN_INPUT = {
    "LIMIT_BAL": 50000, "AGE": 30, "PAY_0": 2, "PAY_2": 2, "PAY_3": 0,
    "BILL_AMT1": 48000, "PAY_AMT1": 1500, "PAY_AMT2": 1200,
    "EDUCATION": 2, "MARRIAGE": 2, "SEX": 2,
}
GOLDEN_PROBABILITY = 0.8940719913890401
GOLDEN_LOG_ODDS = 2.133026599884033


@pytest.fixture(scope="module")
def svc():
    return ModelService()


def test_feature_order_matches_spec(svc):
    assert svc.feature_names == EXPECTED_ORDER


def test_golden_prediction_is_frozen(svc):
    r = svc.decide(GOLDEN_INPUT, threshold=0.5)
    assert r["probability"] == pytest.approx(GOLDEN_PROBABILITY, abs=1e-9)
    assert r["log_odds"] == pytest.approx(GOLDEN_LOG_ODDS, abs=1e-5)
    assert r["decision"] == "decline"


def test_probability_is_sigmoid_of_log_odds(svc):
    r = svc.score(GOLDEN_INPUT)
    assert r["probability"] == pytest.approx(1 / (1 + math.exp(-r["log_odds"])), abs=1e-12)


def test_contributions_plus_base_equal_log_odds(svc):
    # Requesting all features back, TreeSHAP contributions + base must sum to the margin.
    r = svc.score(GOLDEN_INPUT, top_k=len(EXPECTED_ORDER))
    total = r["base_log_odds"] + sum(c["contribution"] for c in r["contributions"])
    assert total == pytest.approx(r["log_odds"], abs=1e-4)


def test_top_k_limits_and_sorts_contributions(svc):
    r = svc.score(GOLDEN_INPUT, top_k=3)
    assert len(r["contributions"]) == 3
    mags = [abs(c["contribution"]) for c in r["contributions"]]
    assert mags == sorted(mags, reverse=True)
    # PAY_0 (recent delinquency) is the dominant risk driver for this applicant.
    assert r["contributions"][0]["feature"] == "PAY_0"


def test_paying_on_time_lowers_risk(svc):
    low = dict(GOLDEN_INPUT)
    low.update({"PAY_0": -1, "PAY_2": -1, "PAY_3": -1})
    assert svc.score(low)["probability"] < svc.score(GOLDEN_INPUT)["probability"]


def test_missing_feature_raises(svc):
    bad = dict(GOLDEN_INPUT)
    del bad["PAY_0"]
    with pytest.raises(ValueError):
        svc.score(bad)


def test_non_numeric_value_raises(svc):
    bad = dict(GOLDEN_INPUT)
    bad["AGE"] = "old"
    with pytest.raises(ValueError):
        svc.score(bad)


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.5])
def test_invalid_threshold_raises(svc, threshold):
    with pytest.raises(ValueError):
        svc.decide(GOLDEN_INPUT, threshold=threshold)


def test_decision_flips_with_threshold(svc):
    assert svc.decide(GOLDEN_INPUT, threshold=0.5)["decision"] == "decline"
    assert svc.decide(GOLDEN_INPUT, threshold=0.95)["decision"] == "approve"
