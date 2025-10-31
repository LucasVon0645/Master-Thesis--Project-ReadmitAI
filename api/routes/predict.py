import json
from functools import lru_cache
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, Body

from api.routes.utils import _to_dict_list
from api.schemas import BatchPayload, PredictionEnvelope
from api.services.prediction import ModelPrediction
from pathlib import Path

# ----------------------- load payload examples -----------------------
EXAMPLE_RESPONSE_PATH = Path(__file__).parent / ".." / "docs" / "example_payload_predict.json"
with EXAMPLE_RESPONSE_PATH.open("r") as f:
    example_payload_predict = json.load(f)
    
# ----------------------- load response examples -----------------------
EXAMPLE_RESPONSE_PATH = Path(__file__).parent / ".." / "docs" / "example_response_predict.json"
with EXAMPLE_RESPONSE_PATH.open("r") as f:
    example_response_predict = json.load(f)


# ----------------------- define router and endpoints -----------------------
router = APIRouter(prefix="/predict", tags=["predict"])


@lru_cache(maxsize=1)
def get_service() -> ModelPrediction:
    """
    Reuse a single ModelPrediction instance per process.
    (Loads model/scaler lazily on first use.)
    """
    return ModelPrediction()


@router.post(
    "",
    response_model=PredictionEnvelope,
    summary="Batch prediction for patient data",
    description="Receives patient-related tables and returns model predictions.",
    responses=example_response_predict
)
def predict(
    payload: BatchPayload = Body(..., description="Batched patient data for prediction.", example=example_payload_predict),
    svc: ModelPrediction = Depends(get_service),
) -> PredictionEnvelope:
    """
    Batch prediction endpoint.

    **Request body:**  
    Provide patient-related tables as JSON arrays (`admissions`, `diagnoses`, `icu_stays`, `procedures`, `prescriptions`, `patients`, `targets`).  
    Each table enforces required columns internally.

    **Returns:**  
    A `PredictionEnvelope` object with predictions and optional metrics.
    """
    try:
        keys_payload = tuple(BatchPayload.model_fields.keys())
        rows_dict: Dict[str, List[Dict[str, Any]]] = {k: _to_dict_list(getattr(payload, k, [])) for k in keys_payload}

        result = svc.predict(rows_dict)  # returns the envelope dict
        
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
