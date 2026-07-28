"""FastAPI backend for heart failure risk prediction and OMOP CDM conversion."""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data_loader import (
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
    HeartFailureDataLoader,
)
from explainability import patient_risk_factors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

app = FastAPI(
    title="Heart Failure Risk Prediction API",
    description=(
        "Acute heart failure risk scoring with false-alarm reduction and "
        "OMOP CDM v5.4 transformation for clinical interoperability."
    ),
    version="1.0.0",
)


class PatientFeatures(BaseModel):
    """Raw clinical measurements for a single patient."""

    age: float = Field(..., ge=0, le=120, description="Patient age in years.")
    anaemia: int = Field(0, ge=0, le=1, description="Decrease of red blood cells (0/1).")
    creatinine_phosphokinase: float = Field(..., ge=0, description="CPK enzyme level (mcg/L).")
    diabetes: int = Field(0, ge=0, le=1, description="Diabetes present (0/1).")
    ejection_fraction: float = Field(..., ge=0, le=100, description="Ejection fraction (%).")
    high_blood_pressure: int = Field(0, ge=0, le=1, description="Hypertension present (0/1).")
    platelets: float = Field(..., ge=0, description="Platelets in blood (kiloplatelets/mL).")
    serum_creatinine: float = Field(..., ge=0, description="Serum creatinine (mg/dL).")
    serum_sodium: float = Field(..., ge=0, description="Serum sodium (mEq/L).")
    sex: int = Field(0, ge=0, le=1, description="Biological sex (0=female, 1=male).")
    smoking: int = Field(0, ge=0, le=1, description="Smoker (0/1).")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 65,
                "anaemia": 0,
                "creatinine_phosphokinase": 146,
                "diabetes": 0,
                "ejection_fraction": 20,
                "high_blood_pressure": 1,
                "platelets": 162000,
                "serum_creatinine": 1.3,
                "serum_sodium": 129,
                "sex": 1,
                "smoking": 1,
            }
        }
    }


class RiskFactor(BaseModel):
    """A single SHAP-derived risk contributor."""

    feature: str
    shap_value: float
    feature_value: float


class PredictionResponse(BaseModel):
    """Risk prediction output with clinical transparency."""

    risk_probability: float = Field(..., description="Predicted probability of an acute event.")
    risk_label: int = Field(..., description="Binary risk flag (1=high risk).")
    threshold: float = Field(..., description="Decision threshold applied.")
    top_risk_factors: list[RiskFactor] = Field(..., description="Top 3 SHAP risk drivers.")


class OmopConversionResponse(BaseModel):
    """OMOP CDM v5.4 Person and Measurement records."""

    person: list[dict[str, Any]]
    measurement: list[dict[str, Any]]


def _build_feature_vector(
    patient: PatientFeatures,
    feature_names: list[str],
    scaler: Any,
) -> np.ndarray:
    """Engineer and scale a single patient into the model's feature space."""
    loader = HeartFailureDataLoader(data_dir=PROJECT_ROOT / "data")
    raw = pd.DataFrame([patient.model_dump()])
    engineered = loader._engineer_features(raw)

    feature_cols = list(CONTINUOUS_FEATURES) + list(BINARY_FEATURES) + ["age_group"]
    features = engineered[feature_cols].copy()
    features = pd.get_dummies(features, columns=["age_group"], prefix="age_group", dtype=float)

    # Align columns to the training feature space (add missing dummy columns as 0).
    for name in feature_names:
        if name not in features.columns:
            features[name] = 0.0
    features = features[feature_names]

    continuous_cols = [c for c in feature_names if c not in BINARY_FEATURES]
    features[continuous_cols] = scaler.transform(features[continuous_cols].astype(float))
    return features.to_numpy()


@lru_cache(maxsize=1)
def load_artifact(model_path: str = str(DEFAULT_MODEL_PATH)) -> dict[str, Any]:
    """Load and cache the trained model artifact."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{path}'. Run `python src/train.py` first."
        )
    with path.open("rb") as handle:
        return pickle.load(handle)


@app.get("/", tags=["health"])
def root() -> dict[str, str]:
    """Service metadata and health hint."""
    return {
        "service": "Heart Failure Risk Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["health"])
def health() -> dict[str, Any]:
    """Report readiness and whether the model artifact is loadable."""
    model_ready = DEFAULT_MODEL_PATH.exists()
    return {"status": "ok", "model_loaded": model_ready}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(patient: PatientFeatures, threshold: float = 0.5) -> PredictionResponse:
    """Predict acute heart failure risk with top-3 SHAP risk factors."""
    try:
        artifact = load_artifact()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    model = artifact["model"]
    feature_names: list[str] = artifact["feature_names"]
    scaler = artifact["scaler"]

    vector = _build_feature_vector(patient, feature_names, scaler)
    probability = float(model.predict_proba(vector)[0, 1])
    label = int(probability >= threshold)

    factors = patient_risk_factors(model, vector, feature_names, top_k=3)

    return PredictionResponse(
        risk_probability=probability,
        risk_label=label,
        threshold=threshold,
        top_risk_factors=[RiskFactor(**f) for f in factors],
    )


@app.post("/convert-omop", response_model=OmopConversionResponse, tags=["interoperability"])
def convert_omop(patients: list[PatientFeatures]) -> OmopConversionResponse:
    """Transform a batch of patient records into OMOP CDM v5.4 tables."""
    if not patients:
        raise HTTPException(status_code=400, detail="At least one patient record is required.")

    loader = HeartFailureDataLoader(data_dir=PROJECT_ROOT / "data")
    df = pd.DataFrame([p.model_dump() for p in patients])
    omop = loader.to_omop_cdm(df)

    return OmopConversionResponse(
        person=omop.person.to_dict(orient="records"),
        measurement=omop.measurement.to_dict(orient="records"),
    )


def main() -> None:
    """Launch the API with uvicorn."""
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
