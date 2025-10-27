# api/schemas.py
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


Row = Dict[str, Any]


class BatchPayload(BaseModel):
    """
    Send your five tables as JSON arrays of row dicts.
    Column validation is enforced by the service using required_cols.
    """
    admissions_diagnoses: List[Row] = Field(default_factory=list)
    icu_stays: List[Row] = Field(default_factory=list)
    procedures: List[Row] = Field(default_factory=list)
    prescriptions: List[Row] = Field(default_factory=list)
    patients: List[Row] = Field(default_factory=list)


class PredictionBody(BaseModel):
    pred_probs: List[float]
    pred_labels: List[int]
    attention_scores: Optional[List[List[float]]] = None
    hadm_ids: Optional[List[Any]] = None


class PredictionMetadata(BaseModel):
    model_name: str
    number_of_predictions: int
    timestamp: str


class PredictionEnvelope(BaseModel):
    prediction: PredictionBody
    metadata: PredictionMetadata
