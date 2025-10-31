from __future__ import annotations
import os
import io
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd

from app.api_client import predict_batch, healthcheck, ApiError
from app.utils import build_predictions_dataframe, format_percentage, plot_probability_distribution, read_csv_to_dataframe, plot_confusion_matrix
from recurrent_health_events_prediction.training.utils import plot_calibration_curve

st.set_page_config(page_title="Readmission Predictions", layout="wide", page_icon=":hospital:")

# --- Sidebar: upload + actions ---
with st.sidebar.expander("Backend"):
    api_ok = False
    try:
        hc = healthcheck()
        if hc.get("status") == "ok":
            st.success(f"API is reachable")
        api_ok = True
    except ApiError as e:
        st.error("API is not reachable")

# File uploader
st.sidebar.header("Upload CSVs")

admission_file = st.sidebar.file_uploader(
    "Admissions File",
    type=["csv"],
    accept_multiple_files=False,
)

diagnoses_file = st.sidebar.file_uploader(
    "Diagnoses File",
    type=["csv"],
    accept_multiple_files=False,
)

icu_stays_file = st.sidebar.file_uploader(
    "ICU Stays File",
    type=["csv"],
    accept_multiple_files=False,
)

patients_file = st.sidebar.file_uploader(
    "Patients File",
    type=["csv"],
    accept_multiple_files=False,
)

procedures_file = st.sidebar.file_uploader(
    "Procedures File",
    type=["csv"],
    accept_multiple_files=False,
)

prescriptions_file = st.sidebar.file_uploader(
    "Prescriptions File",
    type=["csv"],
    accept_multiple_files=False,
)

targets_file = st.sidebar.file_uploader(
    "True Targets File (optional)",
    type=["csv"],
    accept_multiple_files=False,
)

uploaded_files = all([
    admission_file,
    diagnoses_file,
    icu_stays_file,
    patients_file,
    procedures_file,
    prescriptions_file,
])

run_btn = st.sidebar.button("Run predictions", use_container_width=True, disabled=not (uploaded_files and api_ok))
st.sidebar.caption("After uploading CSVs, click **Run predictions** to call the API.")

if "admissions_df" not in st.session_state:
    st.session_state.admissions_df = pd.DataFrame()
if "diagnoses_df" not in st.session_state:
    st.session_state.diagnoses_df = pd.DataFrame()
if "icu_stays_df" not in st.session_state:
    st.session_state.icu_stays_df = pd.DataFrame()
if "patients_df" not in st.session_state:
    st.session_state.patients_df = pd.DataFrame()
if "procedures_df" not in st.session_state:
    st.session_state.procedures_df = pd.DataFrame()
if "prescriptions_df" not in st.session_state:
    st.session_state.prescriptions_df = pd.DataFrame()
if "targets_df" not in st.session_state:
    st.session_state.targets_df = None
if "all_predictions_df" not in st.session_state:
    st.session_state.all_predictions_df = pd.DataFrame()
if "metrics_available" not in st.session_state:
    st.session_state.metrics_available = False

if uploaded_files:
    st.session_state.admissions_df = read_csv_to_dataframe(admission_file)
    st.session_state.diagnoses_df = read_csv_to_dataframe(diagnoses_file)
    st.session_state.icu_stays_df = read_csv_to_dataframe(icu_stays_file)
    st.session_state.patients_df = read_csv_to_dataframe(patients_file)
    st.session_state.procedures_df = read_csv_to_dataframe(procedures_file)
    st.session_state.prescriptions_df = read_csv_to_dataframe(prescriptions_file)
    if targets_file:
        st.session_state.targets_df = read_csv_to_dataframe(targets_file)
    else:
        st.session_state.targets_df = None

if run_btn and uploaded_files and api_ok:
        try:
            response_data = predict_batch(admissions_df=st.session_state.admissions_df,
                                        diagnoses_df=st.session_state.diagnoses_df,
                                        icu_stays_df=st.session_state.icu_stays_df,
                                        patients_df=st.session_state.patients_df,
                                        procedures_df=st.session_state.procedures_df,
                                        prescriptions_df=st.session_state.prescriptions_df,
                                        targets_df=st.session_state.targets_df)
            predictions = response_data.get("prediction", {})
            metadata = response_data.get("metadata", {})
            metrics = response_data.get("metrics", {})
            st.session_state.all_predictions_df = build_predictions_dataframe(predictions)
            st.session_state.metrics_available = metrics["accuracy"] is not None if metrics else False
            st.success(f"Got predictions for {metadata['number_of_predictions']} patients.")
        except ApiError as e:
            st.error(str(e))

# --- Main content ---
st.title("Patient Readmission Dashboard")

if st.session_state.all_predictions_df is None or st.session_state.all_predictions_df.empty:
    st.info("Upload CSVs in the left sidebar and click **Run predictions** to see results.")
else:
    df = st.session_state.all_predictions_df

    tab_all_preds, tab_overview, tab_model_performance = st.tabs(
        ["Model Predictions Table", "Cohort Overview", "Model Performance"]
    )

    with tab_all_preds:
        st.subheader("Model Predictions Table")
        st.dataframe(df, use_container_width=True)
    with tab_overview:
        st.subheader("Cohort Overview")
        st.write("Total Patients:", df["SUBJECT_ID"].nunique())
        st.write("Total Readmissions:", df[df["True Outcome"] == "Readmitted"].shape[0])

        fig = plot_probability_distribution(df)
        st.plotly_chart(fig, use_container_width=True)
    with tab_model_performance:
        st.subheader("Model Performance")
        if not st.session_state.metrics_available:
            st.info("True outcome labels were not provided; model performance metrics cannot be computed.")
        else:
            cols = st.columns(3)
            keys = list(metrics.keys())

            for i, k in enumerate(keys):
                if k == "confusion_matrix":
                    continue
                if k == "roc_curve":
                    k = "ROC AUC"
                    st.metric(label=k, value=format_percentage(metrics["roc_auc"]))
                else:
                    with cols[i % 3]:
                        st.metric(label=k.replace("_", " ").title(), value=format_percentage(metrics[k]))

            st.write("### Confusion Matrix")
            cm = metrics.get("confusion_matrix", {})
            cm = np.array(cm)
            fig = plot_confusion_matrix(cm, class_names=["Not Readmitted", "Readmitted"])
            st.plotly_chart(fig)

            st.write("### Calibration Curve")
            labels = df["True Outcome"].map({"Readmitted": 1, "Not Readmitted": 0}).to_numpy()
            fig = plot_calibration_curve(labels, df["Readmission Prob."], show_plot=False)
            st.plotly_chart(fig)
