from __future__ import annotations
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import json
import math
import streamlit as st

def sidebar_file_uploads(st) -> Tuple[Dict[str, Any], Dict[str, bool]]:
    st.sidebar.header("Upload CSVs 📂")

    st.sidebar.subheader("Admissions File 🚑")
    admission_file = st.sidebar.file_uploader(
        "Required: SUBJECT_ID, HADM_ID, ADMITTIME, DISCHTIME, ADMISSION_TYPE, INSURANCE, ETHNICITY, DISCHARGE_LOCATION",
        type=["csv"],
        help="Should contain admission records for each patient.",
        accept_multiple_files=False,
    )

    st.sidebar.subheader("Diagnoses File 🩺")
    diagnoses_file = st.sidebar.file_uploader(
        "Required: SUBJECT_ID, HADM_ID, ICD9_CODE",
        type=["csv"],
        help="Should contain ICD diagnosis codes linked to each admission.",
        accept_multiple_files=False,
    )

    st.sidebar.subheader("ICU Stays File 🛏️")
    icu_stays_file = st.sidebar.file_uploader(
        "Required: SUBJECT_ID, HADM_ID, INTIME, OUTTIME",
        type=["csv"],
        help="Should contain ICU stay records linked to each admission.",
        accept_multiple_files=False,
    )

    st.sidebar.subheader("Patients File 🤒")
    patients_file = st.sidebar.file_uploader(
        "Required: SUBJECT_ID, HADM_ID, DOB, GENDER",
        type=["csv"],
        help="Should contain patient demographic information.",
        accept_multiple_files=False,
    )

    st.sidebar.subheader("Prescriptions File 💊")
    prescriptions_file = st.sidebar.file_uploader(
        "Required: SUBJECT_ID, HADM_ID, DRUG",
        type=["csv"],
        help="Should contain prescription records linked to each admission.",
        accept_multiple_files=False,
    )

    st.sidebar.subheader("Procedures File 💉")
    procedures_file = st.sidebar.file_uploader(
        "Required: SUBJECT_ID, HADM_ID, ICD9_CODE",
        type=["csv"],
        help="Should contain procedure records linked to each admission.",
        accept_multiple_files=False,
    )

    st.sidebar.subheader("True Targets File (optional) 🎯")
    targets_file = st.sidebar.file_uploader(
        "Required: SUBJECT_ID, HADM_ID, READMISSION_30_DAYS",
        type=["csv"],
        help="Should contain true readmission outcomes for evaluation.",
        accept_multiple_files=False,
    )
    
    st.sidebar.caption("After uploading CSVs, click **Run predictions** to call the API.")

    uploaded_files = all([
        admission_file,
        diagnoses_file,
        icu_stays_file,
        patients_file,
        procedures_file,
        prescriptions_file,
    ])
    
    files = dict([
        ("admission_file", admission_file),
        ("diagnoses_file", diagnoses_file),
        ("icu_stays_file", icu_stays_file),
        ("patients_file", patients_file),
        ("procedures_file", procedures_file),
        ("prescriptions_file", prescriptions_file),
        ("targets_file", targets_file),
    ])
    
    return files, uploaded_files

def initialize_session_state_vars(session_state):
    if "admissions_df" not in session_state:
        session_state.admissions_df = pd.DataFrame()
    if "diagnoses_df" not in session_state:
        session_state.diagnoses_df = pd.DataFrame()
    if "icu_stays_df" not in session_state:
        session_state.icu_stays_df = pd.DataFrame()
    if "patients_df" not in session_state:
        session_state.patients_df = pd.DataFrame()
    if "procedures_df" not in session_state:
        session_state.procedures_df = pd.DataFrame()
    if "prescriptions_df" not in session_state:
        session_state.prescriptions_df = pd.DataFrame()
    if "targets_df" not in session_state:
        session_state.targets_df = None
    if "all_predictions_df" not in session_state:
        session_state.all_predictions_df = pd.DataFrame()
    if "metrics_available" not in session_state:
        session_state.metrics_available = False
    if "metrics_dict" not in session_state:
        session_state.metrics_dict = {}
    if "metadata_dict" not in session_state:
        session_state.metadata_dict = {}

def populate_session_state_from_files(files: Dict[str, Any], st_obj) -> None:
    """Read uploaded CSV files and populate Streamlit `session_state`.

    Parameters
    - files: dict with keys "admission_file", "diagnoses_file", "icu_stays_file",
      "patients_file", "procedures_file", "prescriptions_file", "targets_file".
      Values are the uploaded file-like objects returned by Streamlit's
      `file_uploader`.
    - st_obj: the `streamlit` module (or an object exposing `session_state`).

    This function is intentionally idempotent: it only overwrites session state
    entries when an uploaded file is provided for the corresponding key.
    """
    # admissions_file
    admissions = files.get("admission_file")
    if admissions:
        st_obj.session_state.__setitem__("admissions_df", read_csv_to_dataframe(admissions))

    diagnoses = files.get("diagnoses_file")
    if diagnoses:
        st_obj.session_state.__setitem__("diagnoses_df", read_csv_to_dataframe(diagnoses))
    else:
        st_obj.session_state.__setitem__("diagnoses_df", pd.DataFrame())

    # icu_stays_file
    icu_stays = files.get("icu_stays_file")
    if icu_stays:
        st_obj.session_state.__setitem__("icu_stays_df", read_csv_to_dataframe(icu_stays))
    else:
        st_obj.session_state.__setitem__("icu_stays_df", pd.DataFrame())

    # patients_file
    patients = files.get("patients_file")
    if patients:
        st_obj.session_state.__setitem__("patients_df", read_csv_to_dataframe(patients))
    else:
        st_obj.session_state.__setitem__("patients_df", pd.DataFrame())

    # procedures_file
    procedures = files.get("procedures_file")
    if procedures:
        st_obj.session_state.__setitem__("procedures_df", read_csv_to_dataframe(procedures))
    else:
        st_obj.session_state.__setitem__("procedures_df", pd.DataFrame())

    # prescriptions_file
    prescriptions = files.get("prescriptions_file")
    if prescriptions:
        st_obj.session_state.__setitem__("prescriptions_df", read_csv_to_dataframe(prescriptions))
    else:
        st_obj.session_state.__setitem__("prescriptions_df", pd.DataFrame())

    # targets_file is optional
    targets = files.get("targets_file")
    if targets:
        st_obj.session_state.__setitem__("targets_df", read_csv_to_dataframe(targets))
    else:
        st_obj.session_state.__setitem__("targets_df", None)

def read_csv_to_dataframe(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df

def build_predictions_dataframe(predictions: Dict[str, Any]) -> pd.DataFrame:
    if not predictions:
        return pd.DataFrame(columns=[
            "SUBJECT_ID", "HADM_ID", "Readmission Prob.", "Percentile", "Rank"
        ])

    # Base dictionary
    data_dict = {
        "SUBJECT_ID": predictions["subject_ids"],
        "HADM_ID": predictions["hadm_ids"],
        "Readmission Prob.": predictions["pred_probs"],
    }

    # Optional true label mapping
    if predictions.get("true_labels") is not None:
        true_labels: list = predictions["true_labels"]
        data_dict["True Outcome"] = [
            "Readmitted" if x == 1 else "Not Readmitted" for x in true_labels
        ]

    df = pd.DataFrame(data_dict)

    # --- New columns ---
    # Rank: 1 = highest risk
    df["Rank"] = df["Readmission Prob."].rank(ascending=False, method="min").astype(int)

    # Percentile: relative position (higher = riskier)
    df["Percentile"] = df["Readmission Prob."].rank(pct=True, ascending=True) * 100
    df["Percentile"] = df["Percentile"].round(1)

    # Sort by probability descending for readability
    df = df.sort_values(by="Readmission Prob.", ascending=False).reset_index(drop=True)

    return df

def summarize_predictions(df: pd.DataFrame) -> Dict[str, Any]:
    n_patients = df["SUBJECT_ID"].nunique()
    n_readmitted = df[df["True Outcome"] == "Readmitted"].shape[0]
    summary = {
        "n_patients": n_patients,
        "n_readmitted": n_readmitted,
    }
    return summary

def plot_probability_distribution(df: pd.DataFrame):
    if 'True Outcome'  in df.columns:
        fig = px.histogram(
            df,
            x="Readmission Prob.",
            color="True Outcome",
            nbins=50,
            title="Distribution of Readmission Probabilities by True Outcome",
            labels={"Readmission Prob.": "Predicted Readmission Probability"},
            height=400,
        )

        fig.update_layout(bargap=0.1, yaxis_title="Count")

        return fig
    
    fig = px.histogram(
        df,
        x="Readmission Prob.",
        nbins=50,
        title="Distribution of Readmission Probabilities",
        labels={"Readmission Prob.": "Predicted Readmission Probability"},
        height=400,
    )

    fig.update_layout(bargap=0.1, yaxis_title="Count")

    return fig

def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: list[str],
):
    # Normalize if you want percentages (optional)
    # conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)

    fig = go.Figure(
        data=go.Heatmap(
            z=conf_matrix,
            x=class_names,  # predicted labels
            y=class_names,  # true labels
            colorscale="Blues",
            text=conf_matrix,
            texttemplate="%{text}",
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
            colorbar=dict(title="Count"),
        )
    )

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted label",
        yaxis_title="True label",
        yaxis_autorange="reversed",  # so [0,0] is top-left
        template="plotly_white",
        width=600,
        height=600,
    )

    return fig

def format_percentage(x):
    if x is None:
        return "—"
    # show as percentage with one decimal
    return f"{x*100:.1f}%" if 0 <= x <= 1 else f"{x:.3f}"

def select_patient_data(
    subject_id: int,
    st
) -> Dict[str, pd.DataFrame]:
    
    admissions_df: pd.DataFrame = st.session_state.admissions_df
    diagnoses_df: pd.DataFrame = st.session_state.diagnoses_df
    icu_stays_df: pd.DataFrame = st.session_state.icu_stays_df
    procedures_df: pd.DataFrame = st.session_state.procedures_df
    prescriptions_df: pd.DataFrame = st.session_state.prescriptions_df
    targets_df: Optional[pd.DataFrame] = st.session_state.targets_df
    patients_df: pd.DataFrame = st.session_state.patients_df

    admissions_single_patient_df = admissions_df[
        (admissions_df["SUBJECT_ID"] == subject_id)
    ]
    diagnoses_single_patient_df = diagnoses_df[
        (diagnoses_df["SUBJECT_ID"] == subject_id)
    ]
    icu_stays_single_patient_df = icu_stays_df[
        (icu_stays_df["SUBJECT_ID"] == subject_id)
    ]
    patients_df = patients_df[
        (patients_df["SUBJECT_ID"] == subject_id)
    ]
    procedures_single_patient_df = procedures_df[
        (procedures_df["SUBJECT_ID"] == subject_id)
    ]
    prescriptions_single_patient_df = prescriptions_df[
        (prescriptions_df["SUBJECT_ID"] == subject_id)
    ]
    if targets_df is not None:
        targets_single_patient_df = targets_df[
            (targets_df["SUBJECT_ID"] == subject_id)
        ]
    else:
        targets_single_patient_df = None
        
    results_dict = {
        "admissions_df": admissions_single_patient_df,
        "diagnoses_df": diagnoses_single_patient_df,
        "icu_stays_df": icu_stays_single_patient_df,
        "procedures_df": procedures_single_patient_df,
        "prescriptions_df": prescriptions_single_patient_df,
        "targets_df": targets_single_patient_df,
        "patients_df": patients_df,
    }

    return results_dict

def make_attention_fig(attention_weights, hadm_ids, kind="line"):
    """
    Plot attention weights across the last n admissions.
    
    Parameters
    ----------
    attention_weights : list[float]
        Attention values (length = n). Index 1 corresponds to the earliest
        admission within the observation window.
    hadm_ids : list[str|int]
        All admission IDs. The last n are used, aligned to attention_weights.
    kind : {"bar","line"}
        Choose a bar chart or a line chart with markers.
    """
    attention_weights = list(filter(lambda w: w > 0, attention_weights))
    n = len(attention_weights)
    if n == 0:
        raise ValueError("attention_weights is empty.")
    if len(hadm_ids) < n:
        raise ValueError("hadm_ids must be at least as long as attention_weights.")
    
    # Take the last n admissions and align with the attention weights
    hadm_subset = hadm_ids[-n:]
    x_idx = list(range(1, n + 1))  # 1-based indexing on the x-axis

    if kind == "line":
        trace = go.Scatter(
            x=x_idx, y=attention_weights, mode="lines+markers",
            customdata=np.array(hadm_subset),
            hovertemplate=(
                "Admission index: %{x}<br>"
                "HADM_ID: %{customdata}<br>"
                "Attention: %{y:.2f}<extra></extra>"
            ),
        )
    else:  # "bar"
        trace = go.Bar(
            x=x_idx, y=attention_weights,
            customdata=np.array(hadm_subset),
            hovertemplate=(
                "Admission index: %{x}<br>"
                "HADM_ID: %{customdata}<br>"
                "Attention: %{y:.2}<extra></extra>"
            ),
        )

    fig = go.Figure(trace)
    fig.update_layout(
        title="Attention over the last admissions",
        xaxis=dict(
            title=f"Admission index within observation window "
                  f"(1 = first of the last {n} admissions)",
            dtick=1,          # show only integer ticks (… 1, 2, 3, …)
            tick0=1,          # start ticks at 1
            range=[0.5, n + 0.5],  # centers bars/points on integer positions
        ),
        yaxis=dict(title="Attention weight"),
        margin=dict(l=60, r=20, t=50, b=70),
    )
    return fig

def _pretty_key(k: str) -> str:
    s = str(k).replace("_", " ").strip()
    return s[:1].upper() + s[1:]

def _cell_value(v):
    if v is None:
        return "—"
    if isinstance(v, (dict, list, tuple)):
        # compact JSON for nested structures
        try:
            return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(v)
    return v

def show_kv_two_dfs(
    data: dict,
    *,
    title: str | None = "Current Hospital Admission",
    order: list[str] | None = None,
    n_cols: int = 3,
):
    """
    Render a dict as 'Variable / Value' rows split across n side-by-side tables.
    - Keeps Streamlit look (st.dataframe), no code blocks.
    - Evenly splits rows; left gets the extra if odd.
    - Optional 'order' list to prioritize certain keys first.
    """
    if not data:
        st.info("No features available.")
        return

    # Optional ordering of keys (e.g., to surface vitals first)
    items = list(data.items())
    if order:
        order_rank = {k: i for i, k in enumerate(order)}
        items.sort(key=lambda kv: (order_rank.get(kv[0], float("inf")), str(kv[0]).lower()))

    rows = [{"Variable": _pretty_key(k), "Value": _cell_value(v)} for k, v in items]
    df = pd.DataFrame(rows)

    if title:
        st.markdown(f"##### {title}")

    # Split evenly across columns
    n = len(df)
    per_col = math.ceil(n / n_cols)
    cols = st.columns(n_cols, gap="small")

    for i, col in enumerate(cols):
        start = i * per_col
        end = min((i + 1) * per_col, n)
        sub = df.iloc[start:end].reset_index(drop=True)
        with col:
            st.dataframe(
                sub,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Variable": st.column_config.TextColumn("Variable", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="large"),
                },
            )
