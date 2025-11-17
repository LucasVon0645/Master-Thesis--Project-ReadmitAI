import json
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, Body

from api.routes.utils import _to_dict_list
from api.schemas import PredictionPayload, PredictionBatchEnvelope
from api.services.main import get_service
from api.services.prediction import ModelPrediction
from pathlib import Path

# ----------------------- load payload examples -----------------------
EXAMPLE_REQUEST_PATH = (
    Path(__file__).parent / ".." / "docs" / "example_payload_predict.json"
)
with EXAMPLE_REQUEST_PATH.open("r") as f:
    example_payload_predict = json.load(f)

# ----------------------- load response examples -----------------------
EXAMPLE_RESPONSE_PATH = (
    Path(__file__).parent / ".." / "docs" / "example_response_predict.json"
)
with EXAMPLE_RESPONSE_PATH.open("r") as f:
    example_response_predict = json.load(f)


# ----------------------- define router and endpoints -----------------------
router = APIRouter(prefix="/predict", tags=["predict"])


@router.post(
    "",
    response_model=PredictionBatchEnvelope,
    summary="Batch prediction for patient data",
    description="Receives patient-related tables and returns model predictions.",
    responses=example_response_predict,
)
def predict(
    payload: PredictionPayload = Body(
        ...,
        description="Batched patient data for prediction.",
        examples=example_payload_predict,
    ),
    svc: ModelPrediction = Depends(get_service),
) -> PredictionBatchEnvelope:
    """
    Batch prediction endpoint.

    **Request body:**
    Provide patient-related tables as JSON arrays (`admissions`, `diagnoses`, `icu_stays`, `procedures`, `prescriptions`, `patients`, `targets`).
    Each table enforces required columns internally.

    **Returns:**
    A `PredictionEnvelope` object with predictions and optional metrics.
    """
    try:
        keys_payload = tuple(PredictionPayload.model_fields.keys())
        rows_dict: Dict[str, List[Dict[str, Any]]] = {
            k: _to_dict_list(getattr(payload, k, [])) for k in keys_payload
        }

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
