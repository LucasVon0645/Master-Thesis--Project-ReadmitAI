# api/schemas.py
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---- Row models (allow additional columns) ----
class _AllowExtra(BaseModel):
    class Config:
        extra = "allow"   # accept additional columns beyond the required ones

class AdmissionsRow(_AllowExtra):
    HADM_ID: int
    SUBJECT_ID: int
    ADMITTIME: str
    DISCHTIME: str
    ADMISSION_TYPE: str
    INSURANCE: str
    ETHNICITY: str
    DISCHARGE_LOCATION: str

class DiagnosesRow(_AllowExtra):
    HADM_ID: int
    SUBJECT_ID: int
    ICD9_CODE: str

class ICUStayRow(_AllowExtra):
    SUBJECT_ID: int
    HADM_ID: int
    INTIME: str
    OUTTIME: str

class PrescriptionRow(_AllowExtra):
    HADM_ID: int
    SUBJECT_ID: int
    DRUG: str

class ProcedureRow(_AllowExtra):
    HADM_ID: int
    SUBJECT_ID: int
    ICD9_CODE: str

class PatientRow(_AllowExtra):
    SUBJECT_ID: int
    GENDER: str
    DOB: str

class TargetRow(_AllowExtra):
    HADM_ID: int
    SUBJECT_ID: int
    READMISSION_30_DAYS: int  # or bool if you normalize to True/False


class PredictionPayload(BaseModel):
    """
    Container for batched patient data used for model inference.

    Each field is a table represented as an array of row objects.
    Required columns per table are enforced by the row models; extra columns are allowed.
    """

    admissions: List[AdmissionsRow] = Field(
        default_factory=list,
        description="Hospital admissions.",
        title="Admissions (List[AdmissionsRow])",
    )
    diagnoses: List[DiagnosesRow] = Field(
        default_factory=list,
        description="Admission-linked diagnoses.",
        title="Diagnoses (List[DiagnosesRow])",
    )
    icu_stays: List[ICUStayRow] = Field(
        default_factory=list,
        description="ICU stay intervals.",
        title="ICU Stays (List[ICUStayRow])",
    )
    prescriptions: List[PrescriptionRow] = Field(
        default_factory=list,
        description="Medication orders.",
        title="Prescriptions (List[PrescriptionRow])",
    )
    procedures: List[ProcedureRow] = Field(
        default_factory=list,
        description="Procedures performed.",
        title="Procedures (List[ProcedureRow])",
    )
    patients: List[PatientRow] = Field(
        default_factory=list,
        description="Demographics.",
        title="Patients (List[PatientRow])",
    )
    targets: List[TargetRow] | None = Field(
        default=None,
        description="Optional labels/targets for evaluation.",
        title="Targets (List[TargetRow])",
    )

class PredictionBody(BaseModel):
    pred_probs: List[float]
    pred_labels: List[int]
    true_labels: Optional[List[int]] = None
    attention_weights: Optional[List[List[float]]] = None
    hadm_ids: Optional[List[Any]] = None
    subject_ids: Optional[List[Any]] = None

class PredictionMetadata(BaseModel):
    model_name: str
    number_of_predictions: int
    timestamp: str

class PredictionMetrics(BaseModel):
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None

class InputFeatures(BaseModel):
    past: Optional[List[Dict[str, Any]]]
    current: Dict[str, Any]

class PredictionBatchEnvelope(BaseModel):
    prediction: PredictionBody
    metadata: PredictionMetadata
    metrics: PredictionMetrics

class ExplainSinglePatientEnvelope(BaseModel):
    prediction: PredictionBody
    input_features: InputFeatures
    metadata: PredictionMetadata