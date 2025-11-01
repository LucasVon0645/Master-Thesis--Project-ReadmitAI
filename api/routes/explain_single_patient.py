import json
from functools import lru_cache
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, Body

from api.routes.utils import _to_dict_list
from api.schemas import PredictionPayload, ExplainSinglePatientEnvelope
from api.services.main import get_service
from api.services.prediction import ModelPrediction
from pathlib import Path

# ----------------------- load payload examples -----------------------
EXAMPLE_REQUEST_PATH = (
    Path(__file__).parent / ".." / "docs" / "example_payload_single_patient.json"
)
with EXAMPLE_REQUEST_PATH.open("r") as f:
    example_payload_single_patient = json.load(f)

# ----------------------- load response examples -----------------------
EXAMPLE_RESPONSE_PATH = (
    Path(__file__).parent / ".." / "docs" / "example_response_explain_single_patient.json"
)
with EXAMPLE_RESPONSE_PATH.open("r") as f:
    example_response_explain_single_patient = json.load(f)

# ----------------------- define router and endpoints -----------------------
router = APIRouter(prefix="/explain_single_patient", tags=["explain_single_patient"])


@router.post("/", response_model=ExplainSinglePatientEnvelope, summary="Single patient explanation",
             description="Receives patient-related tables and returns model explanation and computed features for a single patient.",
             responses=example_response_explain_single_patient)
def explain_single_patient(
    payload: PredictionPayload = Body(..., description="Single patient explanation.", example=example_payload_single_patient),
    svc: ModelPrediction = Depends(get_service),
) -> ExplainSinglePatientEnvelope:
    """
    Single patient explanation endpoint.

    **Request body:**
    Provide patient-related tables as JSON arrays (`admissions`, `diagnoses`, `icu_stays`, `procedures`, `prescriptions`, `patients`, `targets`).
    Each table enforces required columns internally.

    **Returns:**
    An `ExplainSinglePatientEnvelope` object with prediction and explanation details.
    """
    try:
        keys_payload = tuple(PredictionPayload.model_fields.keys())
        rows_dict: Dict[str, List[Dict[str, Any]]] = {
            k: _to_dict_list(getattr(payload, k, [])) for k in keys_payload
        }

        result = svc.explain_single_patient(rows_dict)  # returns the explanation dict

        return result  # FastAPI will validate/serialize to ExplainSinglePatientEnvelope

    except FileNotFoundError as e:
        # Missing model/scaler artifacts
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        # Schema/validation issues (e.g., required columns missing)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # General exception catch-all
        raise HTTPException(status_code=500, detail=str(e))
