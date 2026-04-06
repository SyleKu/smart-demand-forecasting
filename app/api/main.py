import json
from pathlib import Path
from typing import Any, Optional
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.ensemble import RandomForestRegressor

from app.api.schemas import ForecastRequest, ForecastResponse
from app.core.config import FEATURE_COLUMNS

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / 'models/trained_model.pkl'
METRICS_PATH = BASE_DIR / 'reports/metrics.json'

model: Optional[RandomForestRegressor] = None
metrics: dict[str, Any] | None = None

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, metrics

    print("Loading model and metrics...")
    model = load_model()
    metrics = load_metrics()

    yield

    print("Shutting down application...")

app = FastAPI(
    title="Smart Demand Forecasting API",
    description="API for energy load forecasting predictions",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    if metrics is None:
        return {
            "model_loaded": model is not None,
            "feature_columns": FEATURE_COLUMNS,
            "metrics_available": False,
        }

    return {
        "model_loaded": model is not None,
        "feature_columns": FEATURE_COLUMNS,
        "metrics_available": True,
        "metrics": metrics,
    }

@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    input_data = pd.DataFrame([request.model_dump()])

    assert isinstance(input_data, pd.DataFrame)

    input_data = input_data.loc[:, FEATURE_COLUMNS]

    missing_features = [col for col in FEATURE_COLUMNS if col not in input_data.columns]
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing_features}",
        )

    input_data = input_data.loc[:, FEATURE_COLUMNS]

    prediction = model.predict(input_data)[0]

    return ForecastResponse(prediction=float(prediction))
