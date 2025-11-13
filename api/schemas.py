# api/schemas.py

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---- Base class allowing extra columns ----
class _AllowExtra(BaseModel):
    class Config:
        extra = "allow"  # accept additional columns beyond the required ones


# --------------------------------------------------------
# Row-level tables (each representing a database-like row)
# --------------------------------------------------------


class AdmissionsRow(_AllowExtra):
    HADM_ID: int = Field(..., description="Hospital admission ID.")
    SUBJECT_ID: int = Field(..., description="Patient ID.")
    ADMITTIME: str = Field(..., description="Admission timestamp.")
    DISCHTIME: str = Field(..., description="Discharge timestamp.")
    ADMISSION_TYPE: str = Field(..., description="Type of admission.")
    INSURANCE: str = Field(..., description="Insurance category.")
    ETHNICITY: str = Field(..., description="Patient ethnicity.")
    DISCHARGE_LOCATION: str = Field(..., description="Discharge destination.")


class DiagnosesRow(_AllowExtra):
    HADM_ID: int = Field(..., description="Hospital admission ID.")
    SUBJECT_ID: int = Field(..., description="Patient ID.")
    ICD9_CODE: str = Field(..., description="Diagnosis code (ICD-9).")


class ICUStayRow(_AllowExtra):
    SUBJECT_ID: int = Field(..., description="Patient ID.")
    HADM_ID: int = Field(..., description="Associated admission ID.")
    INTIME: str = Field(..., description="ICU entry timestamp.")
    OUTTIME: str = Field(..., description="ICU exit timestamp.")


class PrescriptionRow(_AllowExtra):
    HADM_ID: int = Field(..., description="Hospital admission ID.")
    SUBJECT_ID: int = Field(..., description="Patient ID.")
    DRUG: str = Field(..., description="Prescribed drug name.")


class ProcedureRow(_AllowExtra):
    HADM_ID: int = Field(..., description="Hospital admission ID.")
    SUBJECT_ID: int = Field(..., description="Patient ID.")
    ICD9_CODE: str = Field(..., description="Procedure code (ICD-9).")


class PatientRow(_AllowExtra):
    SUBJECT_ID: int = Field(..., description="Patient ID.")
    GENDER: str = Field(..., description="Patient gender.")
    DOB: str = Field(..., description="Date of birth.")


class TargetRow(_AllowExtra):
    HADM_ID: int = Field(..., description="Hospital admission ID.")
    SUBJECT_ID: int = Field(..., description="Patient ID.")
    READMISSION_30_DAYS: int = Field(
        ..., description="30-day readmission indicator (0/1)."
    )


# --------------------------------------------------------
# Prediction input schema
# --------------------------------------------------------


class PredictionPayload(BaseModel):
    """
    Container for batched patient data used for model inference.
    """

    admissions: List[AdmissionsRow] = Field(
        default_factory=list,
        description="Admission records.",
        title="Admissions (List[AdmissionsRow])",
    )
    diagnoses: List[DiagnosesRow] = Field(
        default_factory=list,
        description="Diagnosis records.",
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
        description="Patient demographics.",
        title="Patients (List[PatientRow])",
    )
    targets: Optional[List[TargetRow]] = Field(
        default=None,
        description="Optional labels/targets for evaluation.",
        title="Targets (List[TargetRow])",
    )


# --------------------------------------------------------
# Prediction output structures
# --------------------------------------------------------


class PredictionBody(BaseModel):
    pred_probs: List[float] = Field(..., description="Predicted probabilities.")
    pred_labels: List[int] = Field(..., description="Predicted binary labels.")
    true_labels: Optional[List[int]] = Field(
        None, description="Ground-truth labels, if available."
    )
    attention_weights: Optional[List[List[float]]] = Field(
        None, description="Attention weights per timestep."
    )
    hadm_ids: Optional[List[Any]] = Field(
        None, description="Admission IDs for the predictions."
    )
    subject_ids: Optional[List[Any]] = Field(
        None, description="Patient IDs for the predictions."
    )


class PredictionMetadata(BaseModel):
    model_name: str = Field(..., description="Model identifier.")
    number_of_predictions: int = Field(..., description="Total predictions in batch.")
    timestamp: str = Field(..., description="Prediction timestamp.")
    prob_threshold: float = Field(..., description="Classification threshold.")


class PredictionMetrics(BaseModel):
    accuracy: Optional[float] = Field(None, description="Accuracy score.")
    precision: Optional[float] = Field(None, description="Precision score.")
    recall: Optional[float] = Field(None, description="Recall score.")
    f1_score: Optional[float] = Field(None, description="F1 score.")
    auc_roc: Optional[float] = Field(None, description="ROC-AUC metric.")
    confusion_matrix: Optional[List[List[int]]] = Field(
        None, description="Confusion matrix."
    )


# --------------------------------------------------------
# Explainability models
# --------------------------------------------------------


class InputFeatures(BaseModel):
    past: Optional[List[Dict[str, Any]]] = Field(
        None, description="Past sequence-level features."
    )
    current: Dict[str, Any] = Field(..., description="Current admission features.")


class FeatureAttributionSplit(BaseModel):
    past_attribution: float = Field(..., description="Contribution from past features.")
    current_attribution: float = Field(
        ..., description="Contribution from current features."
    )


class FeatureAttributionRow(BaseModel):
    feature: str = Field(..., description="Feature name.")
    attribution: float = Field(..., description="Contribution score.")


class ExplanationBody(BaseModel):
    current_features_attributions: List[FeatureAttributionRow] = Field(
        ..., description="Attribution scores for current features."
    )
    past_features_attributions: List[FeatureAttributionRow] = Field(
        ..., description="Attribution scores for past features."
    )
    feature_attribution_split: FeatureAttributionSplit = Field(
        ..., description="Split between past and current contributions."
    )


# --------------------------------------------------------
# API envelopes
# --------------------------------------------------------


class PredictionBatchEnvelope(BaseModel):
    prediction: PredictionBody = Field(..., description="Prediction results.")
    metadata: PredictionMetadata = Field(
        ..., description="Metadata for the prediction batch."
    )
    metrics: PredictionMetrics = Field(..., description="Evaluation metrics.")


class ExplainSinglePatientEnvelope(BaseModel):
    explanation: ExplanationBody = Field(
        ..., description="Feature attribution details."
    )
    input_features: InputFeatures = Field(..., description="Patient feature data.")
    metadata: PredictionMetadata = Field(..., description="Prediction metadata.")
