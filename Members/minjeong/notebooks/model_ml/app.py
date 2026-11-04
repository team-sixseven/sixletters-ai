"""HTTP API for Cloud Run. POST /predict accepts one patient's 12 inputs.

Local:  uvicorn app:app --port 8080
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from model_total_process.total_flow import load_model, run_model

model = load_model()
app = FastAPI(title="SCD risk (LR, 12 variables)")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": type(model.named_steps["model"]).__name__}


@app.get("/features")
def features() -> dict:
    return {"features": list(model.feature_names_in_)}


@app.post("/predict")
def predict(x: dict):
    try:
        result = run_model(x)
    except (ValueError, TypeError, KeyError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    return result
