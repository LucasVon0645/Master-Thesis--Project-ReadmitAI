from __future__ import annotations
import json
import os
import requests
import pandas as pd
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Will find the root .env automatically
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
PREDICT_ENDPOINT = os.getenv("PREDICT_ENDPOINT", "/predict")
HEALTH_ENDPOINT = os.getenv("HEALTH_ENDPOINT", "/health")
API_TIMEOUT = float(os.getenv("API_TIMEOUT_SECONDS", "30"))
EXPLAIN_ENDPOINT = os.getenv("EXPLAIN_ENDPOINT", "/explain_single_patient")

class ApiError(Exception):
    pass

def healthcheck() -> Dict[str, Any]:
    url = f"{API_BASE_URL}{HEALTH_ENDPOINT}"
    try:
        r = requests.get(url, timeout=API_TIMEOUT)
        r.raise_for_status()
        return r.json() if "application/json" in r.headers.get("content-type","") else {"status":"ok"}
    except requests.RequestException as e:
        raise ApiError(f"Healthcheck failed: {e}") from e

def predict_batch(
    admissions_df: pd.DataFrame,
    diagnoses_df: pd.DataFrame,
    icu_stays_df: pd.DataFrame,
    patients_df: pd.DataFrame,
    procedures_df: pd.DataFrame,
    prescriptions_df: pd.DataFrame,
    targets_df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    url = f"{API_BASE_URL}{PREDICT_ENDPOINT}"
    
    procedures_df["ICD9_CODE"] = procedures_df["ICD9_CODE"].astype(str)
    
    payload = {
        "admissions": admissions_df.to_dict(orient="records"),
        "diagnoses": diagnoses_df.to_dict(orient="records"),
        "icu_stays": icu_stays_df.to_dict(orient="records"),
        "procedures": procedures_df.to_dict(orient="records"),
        "prescriptions": prescriptions_df.to_dict(orient="records"),
        "patients": patients_df.to_dict(orient="records"),
        "targets": targets_df.to_dict(orient="records") if targets_df is not None else None,
    }
    try:
        payload_json = json.dumps(payload)
        headers = {"Content-Type": "application/json"}
        r = requests.post(url, data=payload_json, timeout=API_TIMEOUT, headers=headers)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        raise ApiError(f"Prediction request failed: {e}") from e

    if not isinstance(data, dict) or "prediction" not in data.keys():
        raise ApiError(f"Unexpected response schema: {data}")

    return data

def explain_single_patient(
    admissions_df: pd.DataFrame,
    diagnoses_df: pd.DataFrame,
    icu_stays_df: pd.DataFrame,
    patients_df: pd.DataFrame,
    procedures_df: pd.DataFrame,
    prescriptions_df: pd.DataFrame,
    targets_df: Optional[pd.DataFrame] = None,
):
    url = f"{API_BASE_URL}{EXPLAIN_ENDPOINT}"

    procedures_df["ICD9_CODE"] = procedures_df["ICD9_CODE"].astype(str)
    
    payload = {
        "admissions": admissions_df.to_dict(orient="records"),
        "diagnoses": diagnoses_df.to_dict(orient="records"),
        "icu_stays": icu_stays_df.to_dict(orient="records"),
        "procedures": procedures_df.to_dict(orient="records"),
        "prescriptions": prescriptions_df.to_dict(orient="records"),
        "patients": patients_df.to_dict(orient="records"),
        "targets": targets_df.to_dict(orient="records") if targets_df is not None else None,
    }
    try:
        payload_json = json.dumps(payload)
        headers = {"Content-Type": "application/json"}
        r = requests.post(url, data=payload_json, timeout=API_TIMEOUT, headers=headers)
        r.raise_for_status()
        data = r.json()
    
    except requests.RequestException as e:
        raise ApiError(f"Explanation request failed: {e}") from e

    if not isinstance(data, dict):
        raise ApiError(f"Unexpected response schema: {data}")

    return data