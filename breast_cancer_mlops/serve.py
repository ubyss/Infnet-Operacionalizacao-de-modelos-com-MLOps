from __future__ import annotations

import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from .model_io import load_model_bundle

model, FEATURE_NAMES, SERVING_RUN_ID = load_model_bundle()

app = FastAPI(title="Breast Cancer Inference", version="1.0.0")


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Mapa nome_da_feature -> valor (mesmas colunas do treino).",
    )


class PredictResponse(BaseModel):
    prediction: int
    probability_malignant: float
    latency_ms: float
    model_run_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model_run_id": SERVING_RUN_ID}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    missing = [c for c in FEATURE_NAMES if c not in req.features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Faltam features: {missing[:10]}{'...' if len(missing) > 10 else ''}",
        )
    row = {c: req.features[c] for c in FEATURE_NAMES}
    X = pd.DataFrame([row])
    t0 = time.perf_counter()
    proba = model.predict_proba(X)[0, 1]
    pred = int(model.predict(X)[0])
    dt_ms = (time.perf_counter() - t0) * 1000
    return PredictResponse(
        prediction=pred,
        probability_malignant=float(proba),
        latency_ms=round(dt_ms, 3),
        model_run_id=SERVING_RUN_ID,
    )


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
