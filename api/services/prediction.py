from importlib import import_module, resources as impresources
from typing import Any, Dict, List, Optional

import joblib
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from recurrent_health_events_prediction import configs
from recurrent_health_events_prediction.data_extraction.utils import (
    assign_charlson_category,
)
from recurrent_health_events_prediction.datasets.HospReadmDataset import (
    HospReadmDataset,
)
from recurrent_health_events_prediction.preprocessing.preprocessors import (
    DataPreprocessorMIMIC,
)
from recurrent_health_events_prediction.utils.general_utils import import_yaml_config


# ----------------------- load YAML configs -----------------------
with open(impresources.files(configs) / "data_config.yaml", encoding="utf-8") as f:
    data_config = yaml.safe_load(f)


class ModelPrediction:
    """
    CPU-only model prediction service.
    - Loads model/scaler lazily.
    - Validates required columns and preprocesses inputs.
    - Builds a PyTorch Dataset and runs batched inference on CPU.
    - Returns probabilities, labels, optional attention, and optional IDs.
    """

    def __init__(self) -> None:
        self.preprocessing_config: Dict[str, Any] = data_config["training_data"][
            "mimic"
        ]
        self.api_config: Dict[str, Any] = data_config["api"]["mimic"]

        # Model config and weights
        self.model_config_path = self.api_config["model_config_path"]
        self.model_config: Dict[str, Any] = import_yaml_config(self.model_config_path)
        self.model_filepath = self.api_config["model_filepath"]
        self._model: Optional[torch.nn.Module] = None

        # Scaler artifact
        self.scaler_filepath = self.api_config["scaler_filepath"]
        self._scaler: Optional[StandardScaler] = None

        self.one_hot_encoder_filepath = self.api_config.get("one_hot_encoder_filepath", None)

        # Required columns (keys should match incoming payload)
        # Expecting: admissions_diagnoses, icu_stays, procedures, prescriptions, patients
        self.required_cols: Dict[str, List[str]] = self.api_config[
            "required_cols"
        ]

        # Column names for IDs and time (used by dataset)
        self.patient_id_col: str = self.preprocessing_config.get(
            "patient_id_col", "SUBJECT_ID"
        )
        self.hosp_id_col: str = self.preprocessing_config.get("hosp_id_col", "HADM_ID")
        self.time_col: str = self.preprocessing_config.get("time_col", "ADMITTIME")

        # Fixed CPU-only
        self._device = torch.device("cpu")

    # ----------------------- lazy properties -----------------------
    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            model_config = self.model_config
            model_params = model_config.get("model_params", {})
            model_class_name: str = model_config["model_class"]

            mod = import_module(
                "recurrent_health_events_prediction.model.RecurrentHealthEventsDL"
            )
            try:
                ModelClass = getattr(mod, model_class_name)
            except AttributeError as e:
                raise ImportError(
                    f"Model class '{model_class_name}' not found in RecurrentHealthEventsDL"
                ) from e

            m: torch.nn.Module = ModelClass(**model_params)
            try:
                state = torch.load(self.model_filepath, map_location="cpu")
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Model file not found: {self.model_filepath}"
                ) from e

            m.load_state_dict(state)
            m.to(self._device).eval()
            self._model = m

        return self._model

    @property
    def scaler(self) -> StandardScaler:
        if self._scaler is None:
            try:
                self._scaler = joblib.load(self.scaler_filepath)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Scaler file not found: {self.scaler_filepath}"
                ) from e
        return self._scaler

    # ----------------------- helpers -----------------------
    def _to_df(self, rows: Optional[List[Dict[str, Any]]], key: str) -> pd.DataFrame:
        """
        Convert list-of-dict rows to a DataFrame, ensuring required columns exist
        and slicing to exactly those columns (like your notebook code).
        """
        df = pd.DataFrame(rows or [])
        if df.empty:
            return pd.DataFrame(columns=self.required_cols[key])

        missing = [c for c in self.required_cols[key] if c not in df.columns]
        if missing:
            raise ValueError(f"{key}: missing required columns: {missing}")

        df = df[self.required_cols[key]].copy()
        return df

    def _create_pytorch_dataset(
        self, preprocessed_df: pd.DataFrame
    ) -> HospReadmDataset:
        model_cfg = self.model_config
        pre_cfg = self.preprocessing_config
        dataset_cfg = {
            "longitudinal_feat_cols": model_cfg["longitudinal_feat_cols"],
            "current_feat_cols": model_cfg["current_feat_cols"],
            "max_seq_len": model_cfg.get("max_sequence_length", 5),
            "no_elective": model_cfg.get("no_elective", True),
            "reverse_chronological_order": model_cfg.get(
                "reverse_chronological_order", True
            ),
            # column names config
            "subject_id_col": pre_cfg.get("patient_id_col", "SUBJECT_ID"),
            "order_col": pre_cfg.get("time_col", "ADMITTIME"),
            "label_col": pre_cfg.get("binary_event_col", "READMISSION_30_DAYS"),
            "next_admt_type_col": pre_cfg.get(
                "next_admt_type_col", "NEXT_ADMISSION_TYPE"
            ),
            "hosp_id_col": pre_cfg.get("hosp_id_col", "HADM_ID"),
            "last_events_only": True,
        }
        return HospReadmDataset(dataframe=preprocessed_df, **dataset_cfg)

    def _preprocess(
        self,
        admissions_df: pd.DataFrame,
        diagnoses_df: pd.DataFrame,
        icu_stays_df: pd.DataFrame,
        procedures_df: pd.DataFrame,
        prescriptions_df: pd.DataFrame,
        patients_df: pd.DataFrame,
        targets_df: Optional[pd.DataFrame] = None,
    ) -> HospReadmDataset:
        """
        Run your preprocessing pipeline and build the dataset for inference.
        """
        # Merge diagnoses into admissions on SUBJECT_ID and HADM_ID
        admissions_df = admissions_df.merge(
            diagnoses_df[["SUBJECT_ID", "HADM_ID", "ICD9_CODE"]],
            on=["SUBJECT_ID", "HADM_ID"],
            how="left"
        )

        # Derive Charlson category on admissions (expects ICD9_CODE present)
        admissions_w_charlson = assign_charlson_category(
            admissions_df, icd_column="ICD9_CODE"
        )

        preprocessor = DataPreprocessorMIMIC(self.preprocessing_config)
        preprocessed_df = preprocessor.preprocess_inference(
            admissions_df=admissions_w_charlson,
            icu_stays_df=icu_stays_df,
            patients_df=patients_df,
            prescriptions_df=prescriptions_df,
            procedures_df=procedures_df,
            one_hot_encoder_filepath=self.one_hot_encoder_filepath,
            scaler=self.scaler,
        )
        target_col = self.preprocessing_config["binary_event_col"]

        # If targets are provided, merge them in preprocessed_df for latter metrics calculation
        if targets_df is not None:
            if target_col in preprocessed_df.columns:
                preprocessed_df = preprocessed_df.drop(columns=[target_col], errors="ignore")
            # Merge targets into preprocessed_df on SUBJECT_ID and HADM_ID
            preprocessed_df = preprocessed_df.merge(
                targets_df,
                on=["SUBJECT_ID", "HADM_ID"],
                how="left"
            )

        dataset = self._create_pytorch_dataset(preprocessed_df)
        return dataset

    # ----------------------- public API -----------------------
    def predict(self, input_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Predict recurrent health events from batched inputs (tables as JSON dictionaries).

        input_data keys (lists of row dicts):
          - "admissions"
          - "diagnoses"
          - "icu_stays"
          - "procedures"
          - "prescriptions"
          - "patients"

        Returns:
          {
            "prediction": {
              "pred_probs": [float, ...],
              "pred_labels": [int, ...],
              "true_labels": [int, ...] | [],  # Empty if target_provided is False
              "attention_scores": [[...], ...] | null,
              "hadm_ids": [int|null, ...] | null,
              "subject_ids": [int|null, ...] | null
            },
            "metadata": {
              "model_name": str,
              "number_of_predictions": int
            },
            "metrics": {
              "auroc": float,
              "confusion_matrix": [[int, int], [int, int]],
              "recall": float,
              "accuracy": float,
              "precision": float
            } | null
          }
        """
        # Quick empty-payload guard
        if not any(
            input_data.get(k)
            for k in (
                "admissions",
                "diagnoses",
                "icu_stays",
                "procedures",
                "prescriptions",
                "patients",
                "targets",
            )
        ):
            return None
        
        single_sample = False # flag for single sample case, updated later
        
        # Convert and validate each table (slice to required cols)
        admissions_df = self._to_df(input_data.get("admissions"), "admissions")
        diagnoses_df = self._to_df(input_data.get("diagnoses"), "diagnoses")
        icu_stays_df = self._to_df(input_data.get("icu_stays"), "icu_stays")
        procedures_df = self._to_df(input_data.get("procedures"), "procedures")
        prescriptions_df = self._to_df(input_data.get("prescriptions"), "prescriptions")
        patients_df = self._to_df(input_data.get("patients"), "patients")
        targets_df = None
        target_provided = False
        if "targets" in input_data and input_data["targets"] is not None:
            target_provided = True
            targets_df = self._to_df(input_data["targets"], "targets")

        dataset = self._preprocess(
            admissions_df=admissions_df,
            diagnoses_df=diagnoses_df,
            icu_stays_df=icu_stays_df,
            procedures_df=procedures_df,
            prescriptions_df=prescriptions_df,
            patients_df=patients_df,
            targets_df=targets_df
        )

        # If preprocessing yields no rows, short-circuit
        if len(dataset) == 0:
            return self._postprocess(
                {
                    "pred_probs": [],
                    "pred_labels": [],
                    "attention_scores": None,
                    "hadm_ids": None,
                    "subject_ids": None,
                }, compute_metrics=False
            )
        
        # Check for single sample case and set flag
        if len(dataset) == 1:
            single_sample = True

        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,  # CPU-only; adjust if you want background workers
            pin_memory=False,  # CPU-only
        )

        # Lists to collect batch results
        all_probs: List[np.ndarray] = []
        all_true_labels: List[np.ndarray] = []
        all_attn: List[np.ndarray] = []

        model = self.model
        prob_threshold: float = float(
            self.model_config.get("probability_threshold", 0.5)
        )
        has_attention = getattr(model, "has_attention", lambda: False)()

        # Batch inference loop
        with torch.no_grad():
            for batch in dataloader:
                # Expected dataset output: (x_current, x_past, mask_past, maybe_targets/ids)
                x_current, x_past, mask_past, true_labels = batch
                # Explicit: ensure CPU tensors
                x_current = x_current.to(self._device)
                x_past = x_past.to(self._device)
                mask_past = mask_past.to(self._device)

                if has_attention:
                    outputs_logits, attention_scores = model(
                        x_current, x_past, mask_past
                    )
                    outputs_logits = outputs_logits.squeeze(-1)
                    if attention_scores is not None:
                        # Make (B, T)
                        att = attention_scores.squeeze(-1).cpu().numpy()
                        all_attn.append(att)
                else:
                    outputs_logits = model(x_current, x_past, mask_past).squeeze(-1)

                probs = torch.sigmoid(outputs_logits).reshape(-1)

                all_probs.append(probs.cpu().numpy())
                all_true_labels.append(true_labels.cpu().numpy())

        # Concatenate all batch results
        attention_weights: list = np.concatenate(all_attn, axis=0).tolist() if all_attn else None
        true_labels: list = np.concatenate(all_true_labels, axis=0).tolist() if target_provided else None
        pred_probs = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
        pred_labels: list = (pred_probs >= prob_threshold).astype(int).tolist() if pred_probs.size > 0 else []
        
        # Convert to list
        pred_probs: list = pred_probs.tolist()

        # Provide IDs back, dataset exposes them aligned with items
        hadm_ids = dataset.hadm_ids
        subject_ids = dataset.subject_ids
        if isinstance(hadm_ids, pd.Series):
            hadm_ids = hadm_ids.astype("Int64").tolist()
        if isinstance(subject_ids, pd.Series):
            subject_ids = subject_ids.astype("Int64").tolist()

        return self._postprocess(
            {
                "pred_probs": pred_probs,
                "pred_labels": pred_labels,
                "true_labels": true_labels,
                "attention_weights": attention_weights,
                "hadm_ids": hadm_ids,
                "subject_ids": subject_ids,
            },
            compute_metrics=target_provided and not single_sample,
        )

    # ----------------------- postprocess -----------------------
    def _postprocess(
        self, pred_results: Dict[str, Any], compute_metrics: bool
    ) -> Dict[str, Any]:
        model_name = self.model_config.get("model_name", "RecurrentHealthEventsDLModel")
        number_of_predictions = len(pred_results.get("pred_probs", []))
        metadata = {
            "model_name": model_name,
            "number_of_predictions": int(number_of_predictions),
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        if compute_metrics:
            true_labels = np.array(pred_results.get("true_labels", []))
            pred_labels = np.array(pred_results.get("pred_labels", []))
            pred_probs = np.array(pred_results.get("pred_probs", []))

            auroc = roc_auc_score(true_labels, pred_probs)
            conf_matrix = confusion_matrix(true_labels, pred_labels).tolist()
            recall = recall_score(true_labels, pred_labels)
            accuracy = accuracy_score(true_labels, pred_labels)
            precision = precision_score(true_labels, pred_labels)
        else:
            auroc = None
            conf_matrix = None
            recall = None
            accuracy = None
            precision = None

        metrics = {
            "auroc": auroc,
            "confusion_matrix": conf_matrix,
            "recall": recall,
            "accuracy": accuracy,
            "precision": precision,
        }

        return {"prediction": pred_results, "metadata": metadata, "metrics": metrics}