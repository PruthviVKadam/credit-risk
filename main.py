"""FastAPI app for the credit-default scorer.

Thin HTTP layer over ModelService:
    GET  /            -> interactive slider UI (static/index.html)
    GET  /health      -> liveness + headline AUC
    GET  /spec        -> feature spec that drives the UI
    GET  /metrics     -> held-out metrics + business threshold curve
    POST /score       -> probability, decision, and SHAP contributions

Run locally:  uvicorn main:app --reload   (UI at /, docs at /docs)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from model_service import ModelService

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Credit Risk Scorer", version="1.0.0")
service = ModelService()


class ScoreRequest(BaseModel):
    features: dict[str, float] = Field(..., description="All features named in /spec.")
    threshold: float = Field(0.5, gt=0.0, lt=1.0, description="Decline if P(default) >= threshold.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "auc_roc": service.metrics["auc_roc"]}


@app.get("/spec")
def spec() -> dict:
    return service.spec


@app.get("/metrics")
def metrics() -> dict:
    return service.metrics


@app.post("/score")
def score(request: ScoreRequest) -> dict:
    try:
        return service.decide(request.features, threshold=request.threshold)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
