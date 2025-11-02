from importlib import import_module, resources as impresources
import json
from typing import Any, Dict, List, Tuple, Optional

import joblib
from scipy import stats
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
)
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from recurrent_health_events_prediction import configs
from recurrent_health_events_prediction.data_extraction.utils import (
    assign_charlson_category,
)
from recurrent_health_events_prediction.datasets.HospReadmDataset import (
    HospReadmDataset,
)
from recurrent_health_events_prediction.model.explainers import explain_deep_learning_model_feat
from recurrent_health_events_prediction.preprocessing.preprocessors import (
    DataPreprocessorMIMIC,
)
from recurrent_health_events_prediction.preprocessing.utils import (
    one_hot_encode_and_drop,
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

        self.one_hot_encoder_filepath = self.api_config.get(
            "one_hot_encoder_filepath", None
        )
        self._one_hot_encoder = None

        self.unscaled_feat_df = None

        self.train_explain_stats_path = self.api_config.get(
            "train_explain_stats_path", None
        )

        self._train_explain_stats: Optional[Dict[str, Any]] = None

        # Required columns (keys should match incoming payload)
        # Expecting: admissions_diagnoses, icu_stays, procedures, prescriptions, patients
        self.required_cols: Dict[str, List[str]] = self.api_config["required_cols"]

        # Column names for IDs and time (used by dataset)
        self.patient_id_col: str = self.preprocessing_config.get(
            "patient_id_col", "SUBJECT_ID"
        )
        self.hosp_id_col: str = self.preprocessing_config.get("hosp_id_col", "HADM_ID")
        self.time_col: str = self.preprocessing_config.get("time_col", "ADMITTIME")
        self.discharge_time_col: str = self.preprocessing_config.get(
            "discharge_time_col", "DISCHTIME"
        )
        self.target_col: str = self.preprocessing_config.get(
            "binary_event_col", "READMISSION_30_DAYS"
        )

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

    @property
    def one_hot_encoder(self) -> OneHotEncoder:
        if self._one_hot_encoder is None:
            try:
                self._one_hot_encoder = joblib.load(self.one_hot_encoder_filepath)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"One-hot encoder file not found: {self.one_hot_encoder_filepath}"
                ) from e
        return self._one_hot_encoder

    @property
    def train_explain_stats(self) -> Dict[str, Any]:
        if self._train_explain_stats is None:
            with open(self.train_explain_stats_path, "r") as f:
                raw = json.load(f)
            stats = {}
            for k, v in raw.items():
                # convert lists back to tensors when appropriate
                if isinstance(v, list):
                    stats[k] = torch.tensor(v)
                else:
                    stats[k] = v
            self._train_explain_stats = stats
        return self._train_explain_stats

    # ----------------------- helpers -----------------------
    def _validate_input_data(self, input_data: Dict[str, List[Dict[str, Any]]]) -> None:
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
            raise ValueError("Input data is empty; no tables provided.")

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

        df = df[self.required_cols[key]]
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

    def _scale_features(self, df):
        if self.scaler is None:
            raise ValueError("Scaler not loaded; cannot scale features.")

        config = self.preprocessing_config
        features_to_scale = config.get("features_to_scale", [])
        features_to_scale = [feat for feat in features_to_scale if feat in df.columns]

        df[features_to_scale] = self.scaler.transform(df[features_to_scale])

        return df

    def _one_hot_encode_features(self, df):
        if self.one_hot_encoder is None:
            raise ValueError("One-hot encoder not loaded; cannot encode features.")

        config = self.preprocessing_config
        one_hot_encode_cols = config.get("features_to_one_hot_encode", [])
        one_hot_cols_to_drop = config.get("one_hot_cols_to_drop", [])

        df, _ = one_hot_encode_and_drop(
            df,
            features_to_encode=one_hot_encode_cols,
            one_hot_cols_to_drop=one_hot_cols_to_drop,
            encoder=self.one_hot_encoder,
            fit_encoder=False,
        )

        return df

    def _get_features_and_target(
        self,
        admissions_df: pd.DataFrame,
        diagnoses_df: pd.DataFrame,
        icu_stays_df: pd.DataFrame,
        procedures_df: pd.DataFrame,
        prescriptions_df: pd.DataFrame,
        patients_df: pd.DataFrame,
        targets_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Run your preprocessing pipeline and build the dataset for inference.
        """
        # Merge diagnoses into admissions on SUBJECT_ID and HADM_ID
        admissions_df = admissions_df.merge(
            diagnoses_df[["SUBJECT_ID", "HADM_ID", "ICD9_CODE"]],
            on=["SUBJECT_ID", "HADM_ID"],
            how="left",
        )

        # Derive Charlson category on admissions (expects ICD9_CODE present)
        admissions_w_charlson = assign_charlson_category(
            admissions_df, icd_column="ICD9_CODE"
        )

        preprocessor = DataPreprocessorMIMIC(self.preprocessing_config)
        preprocessed_df = preprocessor.calculate_features_inference(
            admissions_df=admissions_w_charlson,
            icu_stays_df=icu_stays_df,
            patients_df=patients_df,
            prescriptions_df=prescriptions_df,
            procedures_df=procedures_df,
        )

        # If targets are provided, merge them in preprocessed_df for latter metrics calculation
        if targets_df is not None:
            if self.target_col in preprocessed_df.columns:
                preprocessed_df = preprocessed_df.drop(
                    columns=[self.target_col], errors="ignore"
                )
            # Merge targets into preprocessed_df on SUBJECT_ID and HADM_ID
            preprocessed_df = preprocessed_df.merge(
                targets_df, on=[self.patient_id_col, self.hosp_id_col], how="left"
            )
        return preprocessed_df

    def _transform_input_data(
        self,
        input_data: Dict[str, List[Dict[str, Any]]],
        save_unscaled_features: bool = False,
    ) -> Tuple[HospReadmDataset, bool]:
        """
        Validate and transform input JSON dicts into a PyTorch Dataset for inference.
        Returns the dataset and a flag indicating if targets were provided.
        """
        admissions_df = self._to_df(input_data.get("admissions"), "admissions")
        diagnoses_df = self._to_df(input_data.get("diagnoses"), "diagnoses")
        icu_stays_df = self._to_df(input_data.get("icu_stays"), "icu_stays")
        procedures_df = self._to_df(input_data.get("procedures"), "procedures")
        prescriptions_df = self._to_df(input_data.get("prescriptions"), "prescriptions")
        patients_df = self._to_df(input_data.get("patients"), "patients")
        targets_df = None
        target_provided = False
        if "targets" in input_data and input_data["targets"] is not None and len(input_data["targets"]) > 0:
            target_provided = True
            targets_df = self._to_df(input_data["targets"], "targets")

        preprocessed_df = self._get_features_and_target(
            admissions_df=admissions_df,
            diagnoses_df=diagnoses_df,
            icu_stays_df=icu_stays_df,
            procedures_df=procedures_df,
            prescriptions_df=prescriptions_df,
            patients_df=patients_df,
            targets_df=targets_df,
        )

        preprocessed_df = self._one_hot_encode_features(preprocessed_df)

        if save_unscaled_features:
            self.unscaled_feat_df = preprocessed_df.copy()

        preprocessed_df = self._scale_features(preprocessed_df)

        dataset = self._create_pytorch_dataset(preprocessed_df)

        return dataset, target_provided

    def _get_patient_features(
        self
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        df: pd.DataFrame = self.unscaled_feat_df
        if df is None or df.empty:
            raise ValueError(
                "No DataFrame in HospReadmDataset available for patient explanation."
            )

        # Sort by patient and time
        df = df.sort_values(by=[self.patient_id_col, self.time_col]).reset_index(
            drop=True
        )

        longitudinal_feat_cols = self.model_config["longitudinal_feat_cols"]
        current_feat_cols = self.model_config["current_feat_cols"]

        hosp_id_col = self.hosp_id_col
        patient_id_col = self.patient_id_col
        time_col = self.time_col
        discharge_time_col = self.discharge_time_col

        additional_cols = [patient_id_col, hosp_id_col, time_col, discharge_time_col]

        current_feat_cols = [col.replace("LOG_", "") for col in current_feat_cols]
        longitudinal_feat_cols = [col.replace("LOG_", "") for col in longitudinal_feat_cols]

        curr_df: pd.DataFrame = df.iloc[[-1]][additional_cols + current_feat_cols]
        long_df: pd.DataFrame = df.iloc[0:-1][additional_cols + longitudinal_feat_cols]

        curr_dict = curr_df.to_dict(orient="records")[0]
        long_list_dict = long_df.to_dict(orient="records")

        return curr_dict, long_list_dict

    def _prepare_prediction_results_dict(
        self,
        pred_probs,
        pred_labels,
        true_labels,
        attention_weights,
        hadm_ids,
        subject_ids,
    ):
        return {
            "pred_probs": pred_probs,
            "pred_labels": pred_labels,
            "true_labels": true_labels,
            "attention_weights": attention_weights,
            "hadm_ids": hadm_ids,
            "subject_ids": subject_ids,
        }

    def _get_metadata_dict(
        self, number_of_predictions: int, prob_threshold: Optional[float]
    ) -> Dict[str, Any]:
        model_name = self.model_config.get("model_name", "RecurrentHealthEventsDLModel")
        if prob_threshold is None:
            prob_threshold = float(self.model_config.get("probability_threshold", 0.5))
        return {
            "model_name": model_name,
            "number_of_predictions": int(number_of_predictions),
            "timestamp": pd.Timestamp.now().isoformat(),
            "prob_threshold": prob_threshold,
        }

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

        self._validate_input_data(input_data)
        dataset, target_provided = self._transform_input_data(input_data)

        # If preprocessing yields no rows, short-circuit
        if len(dataset) == 0:
            return self._postprocess_predict(
                {
                    "pred_probs": [],
                    "pred_labels": [],
                    "attention_scores": None,
                    "hadm_ids": None,
                    "subject_ids": None,
                },
                prob_threshold=prob_threshold,
                compute_metrics=False,
            )

        # Check for single sample case and set flag
        single_sample = False
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
        attention_weights: list = (
            np.concatenate(all_attn, axis=0).tolist() if all_attn else None
        )
        true_labels: list = (
            np.concatenate(all_true_labels, axis=0).tolist()
            if target_provided
            else None
        )
        pred_probs = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
        pred_labels: list = (
            (pred_probs >= prob_threshold).astype(int).tolist()
            if pred_probs.size > 0
            else []
        )

        # Convert to list
        pred_probs: list = pred_probs.tolist()

        # Provide IDs back, dataset exposes them aligned with items
        hadm_ids = dataset.hadm_ids
        subject_ids = dataset.subject_ids
        if isinstance(hadm_ids, pd.Series):
            hadm_ids = hadm_ids.astype("Int64").tolist()
        if isinstance(subject_ids, pd.Series):
            subject_ids = subject_ids.astype("Int64").tolist()

        pred_results = self._prepare_prediction_results_dict(
            pred_probs=pred_probs,
            pred_labels=pred_labels,
            true_labels=true_labels,
            attention_weights=attention_weights,
            hadm_ids=hadm_ids,
            subject_ids=subject_ids,
        )

        return self._postprocess_predict(
            pred_results,
            prob_threshold=prob_threshold,
            compute_metrics=target_provided and not single_sample,
        )

    def explain_single_patient(
        self, input_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Predict recurrent health events for a single patient from input tables (as JSON dictionaries).

        input_data keys (row dicts):
          - "admissions"
          - "diagnoses"
          - "icu_stays"
          - "procedures"
          - "prescriptions"
          - "patients"

        Returns:
          {
            "prediction": {
              "pred_probs": [float],
              "pred_labels": [int],
              "attention_scores": [[...]] | null,
              "hadm_ids": [int|null] | null,
              "subject_ids": [int|null] | null
            },
            input_features: {
                "longitudinal": [[...]],
                "current": [...],
            },
            "metadata": {
              "model_name": str,
            },
          }
        """

        self._validate_input_data(input_data)
        dataset, _ = self._transform_input_data(
            input_data, save_unscaled_features=True
        )

        # If preprocessing yields no rows, short-circuit
        if len(dataset) == 0:
            return {
                "prediction": {
                    "pred_probs": [],
                    "pred_labels": [],
                    "attention_scores": None,
                    "hadm_ids": None,
                    "subject_ids": None,
                },
                "input_features": {
                    "longitudinal": [],
                    "current": [],
                },
                "metadata": {
                    "model_name": self.model_config.get(
                        "model_name", "RecurrentHealthEventsDLModel"
                    ),
                },
            }

        if len(dataset) > 1:
            raise ValueError(
                "explain_single_patient expects exactly one patient record."
            )

        model = self.model

        x_current, x_past, mask_past, _ = dataset[0]
        feature_names_curr = self.model_config["current_feat_cols"]
        feature_names_past = self.model_config["longitudinal_feat_cols"]

        fc1_layer = model.classifier_head[0]  # nn.Linear
        stats = self.train_explain_stats

        df_curr, df_past, df_split = explain_deep_learning_model_feat(
            model,
            x_current,
            x_past,
            mask_past,
            feature_names_curr,
            feature_names_past,
            layer_for_split=fc1_layer,
            baseline_strategy="means",
            stats=stats,
            n_steps=64,
            internal_batch_size=16,
        )

        return self._postprocess_explain_single_patient(df_curr, df_past, df_split)

    # ----------------------- postprocess -----------------------
    def _postprocess_predict(
        self, pred_results: Dict[str, Any], prob_threshold: float, compute_metrics: bool
    ) -> Dict[str, Any]:

        # Build metadata
        metadata = self._get_metadata_dict(
            number_of_predictions=len(pred_results.get("pred_probs", [])),
            prob_threshold=prob_threshold,
        )

        # Get performance metrics if requested
        if compute_metrics:
            true_labels = np.array(pred_results.get("true_labels", []))
            pred_labels = np.array(pred_results.get("pred_labels", []))
            pred_probs = np.array(pred_results.get("pred_probs", []))

            auroc = roc_auc_score(true_labels, pred_probs)
            conf_matrix = confusion_matrix(true_labels, pred_labels).tolist()
            recall = recall_score(true_labels, pred_labels)
            accuracy = accuracy_score(true_labels, pred_labels)
            precision = precision_score(true_labels, pred_labels)
            f1 = f1_score(true_labels, pred_labels)
        else:
            auroc = None
            conf_matrix = None
            recall = None
            accuracy = None
            precision = None
            f1 = None

        metrics = {
            "auc_roc": auroc,
            "confusion_matrix": conf_matrix,
            "recall": recall,
            "accuracy": accuracy,
            "precision": precision,
            "f1_score": f1,
        }

        return {"prediction": pred_results, "metadata": metadata, "metrics": metrics}

    def _postprocess_explain_single_patient(
        self, df_curr: pd.DataFrame, df_past: pd.DataFrame, df_split: pd.DataFrame
    ) -> Dict[str, Any]:
        # Build metadata
        metadata = self._get_metadata_dict(
            number_of_predictions=1, prob_threshold=None
        )

        # Get input features
        current_features, past_features = self._get_patient_features()
        input_features = {
            "past": past_features,
            "current": current_features,
        }

        df_curr = df_curr[["feature", "attribution"]].sort_values(by="attribution", ascending=False)
        df_past = df_past[["feature", "attribution"]].sort_values(by="attribution", ascending=False)
        split_dict = df_split[["past_attribution", "current_attribution"]].iloc[0].to_dict()

        return {
            "explanation": {
                "current_features_attributions": df_curr.to_dict(orient="records"),
                "past_features_attributions": df_past.to_dict(orient="records"),
                "feature_attribution_split": split_dict,
            },
            "input_features": input_features,
            "metadata": metadata,
        }
