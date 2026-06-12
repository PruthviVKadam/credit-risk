"""API contract tests using FastAPI's TestClient."""

import pytest
from fastapi.testclient import TestClient

from main import app

VALID = {
    "LIMIT_BAL": 50000, "AGE": 30, "PAY_0": 2, "PAY_2": 2, "PAY_3": 0,
    "BILL_AMT1": 48000, "PAY_AMT1": 1500, "PAY_AMT2": 1200,
    "EDUCATION": 2, "MARRIAGE": 2, "SEX": 2,
}


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "auc_roc" in r.json()


def test_spec_lists_features(client):
    r = client.get("/spec")
    assert r.status_code == 200
    names = [f["name"] for f in r.json()["features"]]
    assert "PAY_0" in names and len(names) == 11


def test_metrics_has_threshold_curve(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "auc_roc" in body
    assert isinstance(body["threshold_curve"], list) and body["threshold_curve"]


def test_score_returns_probability_decision_and_shap(client):
    r = client.post("/score", json={"features": VALID, "threshold": 0.5})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["decision"] in {"approve", "decline"}
    assert 1 <= len(body["contributions"]) <= 8
    assert {"feature", "value", "contribution"} <= body["contributions"][0].keys()


def test_score_decision_flips_with_threshold(client):
    high = client.post("/score", json={"features": VALID, "threshold": 0.5}).json()
    low = client.post("/score", json={"features": VALID, "threshold": 0.95}).json()
    assert high["decision"] == "decline"
    assert low["decision"] == "approve"


def test_score_missing_feature_returns_422(client):
    bad = dict(VALID)
    del bad["PAY_0"]
    r = client.post("/score", json={"features": bad, "threshold": 0.5})
    assert r.status_code == 422


def test_score_threshold_out_of_range_returns_422(client):
    r = client.post("/score", json={"features": VALID, "threshold": 1.5})
    assert r.status_code == 422


def test_index_and_static_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
