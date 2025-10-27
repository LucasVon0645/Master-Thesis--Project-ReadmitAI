from functools import lru_cache
from typing import Dict, List
from fastapi import APIRouter, Depends, HTTPException

from recurrent_health_events_prediction.api.schemas import BatchPayload, PredictionEnvelope
from recurrent_health_events_prediction.api.services.prediction import ModelPrediction


router = APIRouter(prefix="/predict", tags=["predict"])


@lru_cache(maxsize=1)
def get_service() -> ModelPrediction:
    """
    Reuse a single ModelPrediction instance per process.
    (Loads model/scaler lazily on first use.)
    """
    return ModelPrediction()


@router.post("", response_model=PredictionEnvelope)
def predict(payload: BatchPayload, svc: ModelPrediction = Depends(get_service)) -> PredictionEnvelope:
    """
    Batch prediction endpoint.
    Accepts five tables, returns probabilities/labels (+ optional IDs, attention).
    """
    try:
        rows: Dict[str, List[dict]] = {
            "admissions_diagnoses": payload.admissions_diagnoses,
            "icu_stays": payload.icu_stays,
            "procedures": payload.procedures,
            "prescriptions": payload.prescriptions,
            "patients": payload.patients,
        }
        result = svc.predict(rows)  # returns the envelope dict
        return result  # FastAPI will validate/serialize to PredictionEnvelope
    except FileNotFoundError as e:
        # Missing model/scaler artifacts
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        # Schema/validation issues (e.g., required columns missing)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # Avoid leaking internals in prod; log e instead.
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
