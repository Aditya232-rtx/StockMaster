"""FastAPI application exposing the anomaly detection service."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure the project root is importable when the API runs via uvicorn
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator
from starlette.staticfiles import StaticFiles

from .service import InventoryAnomalyService

DATA_DIR = PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "output"
FRONTEND_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Inventory Anomaly Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = InventoryAnomalyService(
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
)


class PredictionRequest(BaseModel):
    """Incoming payload for prediction requests."""

    contamination: float = Field(0.05, ge=0.001, le=0.5)
    n_estimators: int = Field(100, ge=10, le=1000)
    random_state: int = Field(42, ge=0, le=2**31 - 1)
    top_n: int = Field(20, ge=1, le=200)

    @validator("contamination")
    def validate_contamination(cls, value: float) -> float:
        if not 0 < value < 0.5:
            raise ValueError("contamination must be between 0 and 0.5")
        return value


class AnomalyRecord(BaseModel):
    transaction_id: Optional[str]
    date: Optional[str]
    product_sku: Optional[str]
    transaction_type: Optional[str]
    quantity_change: Optional[float]
    location: Optional[str]
    anomaly_score: float
    risk_level: str


class PredictionSummaryModel(BaseModel):
    total_transactions: int
    anomalies_detected: int
    anomaly_rate: float
    risk_distribution: Dict[str, int]


class PredictionResponse(BaseModel):
    summary: PredictionSummaryModel
    top_anomalies: List[AnomalyRecord]


@app.on_event("startup")
async def startup_event() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def serve_frontend() -> HTMLResponse:
    """Serve the micro frontend HTML."""
    if not FRONTEND_DIR.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend entrypoint missing")
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest) -> PredictionResponse:
    """Run the anomaly detection pipeline and return predictions."""
    try:
        top_df, summary = service.generate_predictions(
            contamination=payload.contamination,
            n_estimators=payload.n_estimators,
            random_state=payload.random_state,
            top_n=payload.top_n,
        )
    except Exception as exc:  # pragma: no cover - propagates to caller
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response = PredictionResponse(
        summary=PredictionSummaryModel(**service.serialize_summary(summary)),
        top_anomalies=[
            AnomalyRecord(**record)  # type: ignore[arg-type]
            for record in service.serialize_results(top_df)
        ],
    )
    return response


# Serve additional static assets (JS/CSS) for the frontend if present
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
