from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

from app.api_client import explain_single_patient, predict_batch, healthcheck, ApiError
from app.utils import (
    build_predictions_dataframe,
    format_percentage,
    initialize_session_state_vars,
    make_attention_fig,
    plot_probability_distribution,
    plot_confusion_matrix,
    select_patient_data,
    sidebar_file_uploads,
    populate_session_state_from_files,
)
from recurrent_health_events_prediction.training.utils import plot_calibration_curve
from recurrent_health_events_prediction.visualization.utils import plot_subject_evolution

st.set_page_config(page_title="Hospital Readmission Prediction System", layout="wide", page_icon=":hospital:")

st.title("Hospital Readmission Prediction System")

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

# File uploader (moved into helper for cleanliness)
files, uploaded_files = sidebar_file_uploads(st)

admission_file = files["admission_file"]
diagnoses_file = files["diagnoses_file"]
icu_stays_file = files["icu_stays_file"]
patients_file = files["patients_file"]
procedures_file = files["procedures_file"]
prescriptions_file = files["prescriptions_file"]
targets_file = files["targets_file"]

run_btn = st.sidebar.button("Run predictions", use_container_width=True, disabled=not (uploaded_files and api_ok))

initialize_session_state_vars(st.session_state)

# When uploads are present, read them into session state (centralized helper)
if uploaded_files:
    populate_session_state_from_files(files, st)

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
            st.session_state.metrics_dict = metrics
            st.session_state.metadata_dict = metadata
            st.success(f"Got predictions for {metadata['number_of_predictions']} patients.")
        except ApiError as e:
            st.error(str(e))

if st.session_state.all_predictions_df is None or st.session_state.all_predictions_df.empty:
    st.info("Upload CSVs in the left sidebar and click **Run predictions** to see results.")
else:
    df = st.session_state.all_predictions_df
    metrics = st.session_state.metrics_dict
    metadata = st.session_state.metadata_dict
    
    NAV_OPTIONS = [
        "Model Predictions Table",
        "Cohort Overview",
        "Specific Patient",
        "Model Performance",
    ]

    st.header("Prediction Results")
    tab_all_preds, tab_overview, tab_specific_patient, tab_model_performance = st.tabs(NAV_OPTIONS)

    with tab_all_preds:
        st.subheader("Model Predictions Table")
        st.dataframe(df, use_container_width=True)
    with tab_overview:
        st.subheader("Cohort Overview")
        st.write("Total Patients:", df["SUBJECT_ID"].nunique())
        st.write("Total Readmissions:", df[df["True Outcome"] == "Readmitted"].shape[0])
        fig = plot_probability_distribution(df)
        st.plotly_chart(fig, use_container_width=True)
    with tab_specific_patient:
        st.subheader(f"Results for Specific Patient")

        patient_ids = df["SUBJECT_ID"].unique().tolist()
        options = [None] + patient_ids
        selected_patient_id = st.selectbox(
            "Select Patient SUBJECT_ID",
            options,
            index=0,
            format_func=lambda x: "— Select a patient —" if x is None else str(x),
        )

        btt_explain = st.button("Check Results of Patient", use_container_width=True)
        if btt_explain and not selected_patient_id:
            st.warning("Please select a patient to analyze.")
        if selected_patient_id and btt_explain:
            patient_data = select_patient_data(selected_patient_id, st)
            explanation = explain_single_patient(**patient_data)
            pred_prob_patient = None
            true_label_patient = None
            attention_weights = None

            if explanation:
                st.markdown(f"### Results for Patient {selected_patient_id}")
                prediction = explanation.get("prediction", {})
                if prediction:
                    pred_prob_patient = prediction["pred_probs"][0]
                    true_label_patient = (
                        prediction["true_labels"][0]
                        if prediction["true_labels"] is not None
                        else None
                    )
                    attention_weights = (
                        prediction["attention_weights"][0]
                        if prediction["attention_weights"] is not None
                        else None
                    )

                    patient_results_cols = st.columns(2)
                    with patient_results_cols[0]:
                        st.metric(label="Readmission Probability", value=f"{format_percentage(pred_prob_patient)}")
                    with patient_results_cols[1]:
                        if true_label_patient:
                            st.metric(label="True Outcome", value="Readmitted" if true_label_patient == 1 else "Not Readmitted")
                st.divider()
                input_features = explanation.get("input_features", {})

                if not input_features:
                    st.warning("No input features available for this patient.")
                else:
                    past_features_df = pd.DataFrame(input_features.get("past", [{}]))
                    current_features_dict = input_features.get("current", {})
                    st.markdown("#### Current Hospital Admission")
                    st.write(current_features_dict)
                    st.divider()
                    st.markdown("#### Past Hospital Admissions History")
                    feat_display = past_features_df.columns.tolist()[4:]
                    fig = plot_subject_evolution(
                        past_features_df,
                        selected_patient_id,
                        features_to_plot=feat_display,
                        textposition="auto",
                        extend_time_horizon_by=30,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("Table - Past Admissions"):
                        st.dataframe(past_features_df, use_container_width=True)

                if prediction:
                    if attention_weights:
                        attention_weights = list(filter(lambda w: w > 0, attention_weights))
                        past_obs_window = len(attention_weights)
                        past_hadm_ids = past_features_df["HADM_ID"].tolist()
                        fig = make_attention_fig(
                            attention_weights=attention_weights,
                            hadm_ids=past_hadm_ids,
                            kind="line",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(
                            f"Attention Weights over the observation window of the past {past_obs_window} admissions"
                        )

    with tab_model_performance:
        st.subheader("Performance Metrics")
        if not st.session_state.metrics_available:
            st.info("True outcome labels were not provided; model performance metrics cannot be computed.")
        else:
            metrics = metrics or {}
            cols = st.columns(3)
            keys = list(metrics.keys())
            model_name = metadata.get("model_name", "Unknown Model")
            st.markdown("Model Name: `" + model_name + "`")
            for i, k in enumerate(keys):
                if k == "confusion_matrix":
                    continue
                if k == "roc_curve":
                    k = "ROC AUC"
                    st.metric(label=k, value=format_percentage(metrics["roc_auc"]))
                else:
                    with cols[i % 3]:
                        st.metric(label=k.replace("_", " ").title(), value=format_percentage(metrics[k]))
            st.divider()
            st.write("### Confusion Matrix")
            cm = metrics.get("confusion_matrix", {})
            cm = np.array(cm)
            fig = plot_confusion_matrix(cm, class_names=["Not Readmitted", "Readmitted"])
            st.plotly_chart(fig)
            st.divider()
            st.write("### Calibration Curve")
            labels = df["True Outcome"].map({"Readmitted": 1, "Not Readmitted": 0}).to_numpy()
            fig = plot_calibration_curve(labels, df["Readmission Prob."], show_plot=False)
            st.plotly_chart(fig)
