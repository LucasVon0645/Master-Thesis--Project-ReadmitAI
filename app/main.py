from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

from app.api_client import explain_single_patient, predict_batch, healthcheck, ApiError
from app.home import render_home
from app.plots import (
    make_attention_fig,
    plot_confusion_matrix,
    plot_feature_attributions,
    plot_probability_distribution,
    plot_subject_evolution,
    plot_calibration_curve,
)
from app.utils import (
    get_feature_value_dfs,
    get_mean_training_feature_values,
    build_att_weights_dict,
    build_predictions_dataframe,
    format_percentage,
    get_specific_patient_pred,
    initialize_session_state_vars,
    load_css,
    make_feature_attr_df,
    select_patient_data,
    show_kv_two_dfs,
    sidebar_file_uploads,
    populate_session_state_from_files,
)

st.set_page_config(
    page_title="Hospital Readmission Prediction System",
    layout="wide",
    page_icon=":hospital:",
)

load_css("/workspaces/msc-thesis-recurrent-health-modeling/app/style.css")

st.title("Hospital Readmission Prediction System :hospital:")

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

run_btn = st.sidebar.button(
    "Run predictions",
    use_container_width=True,
    disabled=not (uploaded_files and api_ok),
)

initialize_session_state_vars()

# When uploads are present, read them into session state (centralized helper)
if uploaded_files:
    populate_session_state_from_files(files, st)

# --- Run predictions block (unchanged, just ensure it runs before the UI) ---
if run_btn and uploaded_files and api_ok:
    with st.spinner("Running predictions... Please wait while the model processes your data."):
        try:
            response_data = predict_batch(
                admissions_df=st.session_state.admissions_df,
                diagnoses_df=st.session_state.diagnoses_df,
                icu_stays_df=st.session_state.icu_stays_df,
                patients_df=st.session_state.patients_df,
                procedures_df=st.session_state.procedures_df,
                prescriptions_df=st.session_state.prescriptions_df,
                targets_df=st.session_state.targets_df,
            )
            predictions = response_data.get("prediction", {})
            metadata = response_data.get("metadata", {})
            metrics = response_data.get("metrics", {})
            st.session_state.all_predictions_df = build_predictions_dataframe(predictions)
            st.session_state.att_weights_dict = build_att_weights_dict(predictions)
            st.session_state.metrics_available = metrics["accuracy"] is not None if metrics else False
            st.session_state.metrics_dict = metrics
            st.session_state.metadata_dict = metadata
            st.success(f"Got predictions for {metadata['number_of_predictions']} patients.")
        except ApiError as e:
            st.error(str(e))

# --- Guard: show onboarding message if no data ---
if (
    st.session_state.all_predictions_df is None
    or st.session_state.all_predictions_df.empty
):
    st.info(
        "Upload CSVs in the left sidebar and click **Run predictions** to see results."
    )
    render_home()
    st.stop()

# --- Shorthand handles ---
df = st.session_state.all_predictions_df
metrics = st.session_state.metrics_dict or {}
metadata = st.session_state.metadata_dict or {}

# =========================
# Persistent Navigation
# =========================
NAV_OPTIONS = [
    "Model Predictions Table",
    "Cohort Overview",
    "Specific Patient",
    "Model Performance",
]

if "active_view" not in st.session_state:
    st.session_state.active_view = NAV_OPTIONS[0]

# Use a horizontal radio to mimic tabs but preserve selection on reruns
st.markdown("### Prediction Results")
active_view = st.radio(
    "Select view",
    NAV_OPTIONS,
    index=NAV_OPTIONS.index(st.session_state.active_view),
    horizontal=True,
    label_visibility="collapsed",
    key="active_view",
)

st.divider()

prob_threshold = st.session_state.metadata_dict.get("prob_threshold", 0.5)

# =========================
# View 1: Model Predictions Table
# =========================
if active_view == "Model Predictions Table":
    st.subheader("Model Predictions Table")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"A probability threshold of {prob_threshold:.2f} is used.")

# =========================
# View 2: Cohort Overview
# =========================
elif active_view == "Cohort Overview":
    st.subheader("Cohort Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Patients", f"{df['SUBJECT_ID'].nunique():,}")
    with c2:
        expected_num_admissions = (st.session_state.all_predictions_df["Predicted Outcome"] == "Readmitted").sum()
        st.metric("Expected Readmissions", str(expected_num_admissions))
        st.caption(f"A probability threshold of {prob_threshold:.2f} is used.")
    with c3:
        if 'True Outcome' in df.columns:
            value = f"{df[df['True Outcome'] == 'Readmitted'].shape[0]}"
        else:
            value = "Not Available"
        st.metric("Total Readmissions", value)

    fig = plot_probability_distribution(df, prob_threshold)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# View 3: Specific Patient
# =========================
elif active_view == "Specific Patient":
    st.subheader("Results for Specific Patient")

    # --- Selector + Button layout ---
    sel_col, spacer_col, btn_col = st.columns([3, 0.15, 1])

    patient_ids = df["SUBJECT_ID"].unique().tolist()
    options = [None] + patient_ids

    with sel_col:
        selected_patient_id = st.selectbox(
            "Select Patient SUBJECT_ID",
            options,
            index=options.index(st.session_state.selected_patient_id)
                if st.session_state.selected_patient_id in options
                else 0,
            format_func=lambda x: "— Select a patient —" if x is None else str(x),
            key="selected_patient_id",
            on_change=lambda: st.session_state.update(active_view="Specific Patient"),
        )

    # Spacer for visual separation
    with spacer_col:
        st.write("")  # just creates a small horizontal gap

    with btn_col:
        # CSS: vertically center the button inside its column
        st.markdown(
            """
            <style>
            /* Centers the button in its parent column */
            div[data-testid="column"]:has(button) {
                display: flex;
                align-items: center;
                justify-content: center;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        btt_explain = st.button(
            "Check Results",
            use_container_width=True,
            key="explain_btn",
            on_click=lambda: st.session_state.update(active_view="Specific Patient"),
        )

    st.caption("Tip: changing the selector won't run the model until you click **Check Results**.")
    if btt_explain and not selected_patient_id:
        st.warning("Please select a patient to analyze.")
        st.stop()

    if selected_patient_id and btt_explain:
        patient_data = select_patient_data(selected_patient_id)
        explanation_response = explain_single_patient(**patient_data)

        if not explanation_response:
            st.error("No explanation returned for this patient.")
            st.stop()

        prediction = get_specific_patient_pred(selected_patient_id)
        input_features = explanation_response.get("input_features", {})
        explanation = explanation_response.get("explanation", {})
        pred_prob_patient = None
        true_label_patient = None
        attention_weights = None

        # --- Prediction summary ---
        if prediction:
            pred_prob_patient = prediction["pred_prob"]
            true_label_patient = prediction.get("true_label", None)
            rank = prediction["rank"]
            percentile = prediction["percentile"]

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.metric("Readmission Prob.", format_percentage(pred_prob_patient))
            with kpi2:
                if true_label_patient is not None:
                    st.metric(
                        "True Outcome Readmitted",
                        "Yes" if true_label_patient == 1 else "No",
                    )
            with kpi3:
                st.metric(
                    "Risk Rank",
                    f"{rank}",
                    help="Rank of this patient's readmission risk within the cohort.",
                )
            with kpi4:
                st.metric(
                    "Risk Percentile",
                    f"{percentile:.2f}",
                    help="Percentile of this patient's readmission risk within the cohort.",
                )

        st.divider()

        st.markdown(f"#### Admission Details of Patient {selected_patient_id}")
        if not input_features:
            st.warning("No input features available for this patient.")
            st.stop()

        current_features_dict = input_features.get("current", {})
        past_features_df = pd.DataFrame(input_features.get("past", [{}]))
    
        curr_feat_values_df, past_mean_feat_values_df = get_feature_value_dfs(current_features_dict, past_features_df)
        
        mean_curr_train_df, mean_past_train_df = get_mean_training_feature_values()

        curr_features_att_df = make_feature_attr_df(explanation["current_features_attributions"], curr_feat_values_df, mean_curr_train_df)
        past_features_att_df = make_feature_attr_df(explanation["past_features_attributions"], past_mean_feat_values_df, mean_past_train_df)
        
        curr_overall_attr = explanation["feature_attribution_split"]["current_attribution"]
        past_overall_attr = explanation["feature_attribution_split"]["past_attribution"]

        # Tabs for current vs past admission details
        tab_current, tab_past, tab_feat_attr = st.tabs(["Current Admission", "Past Admissions", "Feature Attributions"])

        with tab_current:
            show_kv_two_dfs(current_features_dict, n_cols=3, title="Current Hospital Admission Details")

        with tab_past:
            st.markdown("##### Past Hospital Admissions Overview")
            feat_display = past_features_df.columns.tolist()[4:]
            fig = plot_subject_evolution(
                past_features_df,
                selected_patient_id,
                features_to_plot=feat_display,
                textposition="auto",
                extend_time_horizon_by=30,
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Table — Past Admissions"):
                st.dataframe(past_features_df, use_container_width=True, hide_index=True)

            # Attention weights
            attention_weights = st.session_state.att_weights_dict.get(selected_patient_id, [])
            if len(attention_weights) > 0:
                attention_weights = [w for w in attention_weights if w > 0]
                if attention_weights:
                    past_obs_window = len(attention_weights)
                    past_hadm_ids = (
                        past_features_df["HADM_ID"].tolist()
                        if "HADM_ID" in past_features_df.columns
                        else []
                    )
                    st.markdown("###### Attention Weights")
                    fig = make_attention_fig(
                        attention_weights=attention_weights,
                        hadm_ids=past_hadm_ids,
                        kind="line",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        f"Attention weights over the observation window of the past {past_obs_window} admissions."
                    )
            else:
                st.info("No attention weights available for this patient.")

        with tab_feat_attr:
            st.markdown("##### Feature Attributions")
            feat_attr_col1, feat_attr_col2 = st.columns(2)
            with feat_attr_col1:
                st.metric(
                    "Overall Attribution — Current Admission",
                    f"{curr_overall_attr:.4f}",
                    help="This score indicates how much the model relied on current admission features for its prediction.",
                )
                fig = plot_feature_attributions(
                    curr_features_att_df,
                    title="Current Admission Feature Attributions - Top 10",
                    top_k=10,
                    feature_col="Feature",
                    attr_col="Attribution",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.expander(
                    "Full Current Admission Feature Attributions", expanded=False
                ).dataframe(curr_features_att_df.style.format({
                            col: "{:.3f}" for col in curr_features_att_df.select_dtypes(include="float").columns
                        }), use_container_width=True, hide_index=True)
            with feat_attr_col2:
                st.metric(
                    "Overall Attribution — Past Admissions",
                    f"{past_overall_attr:.4f}",
                    help="This score indicates how much the model relied on past admission features for its prediction.",
                )
                fig = plot_feature_attributions(
                    past_features_att_df,
                    title="Past Admissions Feature Attributions - Top 10",
                    top_k=10,
                    feature_col="Feature",
                    attr_col="Attribution",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.expander(
                    "Full Past Admissions Feature Attributions", expanded=False
                ).dataframe(past_features_att_df.style.format({
                                col: "{:.3f}" for col in past_features_att_df.select_dtypes(include="float").columns
                            }), use_container_width=True, hide_index=True)
        st.caption(
            "Feature attributions indicate the contribution of each feature to the model's prediction."
        )

# =========================
# View 4: Model Performance
# =========================
elif active_view == "Model Performance":
    st.subheader("Performance Metrics")

    if not st.session_state.metrics_available:
        st.info("True outcome labels were not provided; model performance metrics cannot be computed.")
        st.stop()

    cols = st.columns(3)
    model_name = metadata.get("model_name", "Unknown Model")
    prob_threshold = metadata.get("prob_threshold", 0.5)
    st.markdown("**Probability Threshold**: **" + f"{prob_threshold:.2f}" + "**")
    st.markdown("**Model Name**: `" + model_name + "`")

    # Arrange metrics into cards, skipping arrays/plots
    for i, k in enumerate(list(metrics.keys())):
        if k in ("confusion_matrix", "roc_curve"):
            continue
        with cols[i % 3]:
            st.metric(label=k.replace("_", " ").title(), value=format_percentage(metrics[k]))

    st.divider()
    col1_performance, col2_performance = st.columns(2)

    with col1_performance:
        cm = np.array(metrics.get("confusion_matrix", {}))
        fig = plot_confusion_matrix(cm, class_names=["Not Readmitted", "Readmitted"])
        st.plotly_chart(fig, use_container_width=True)

    with col2_performance:
        if "True Outcome" in df.columns and "Readmission Prob." in df.columns:
            labels = df["True Outcome"].map({"Readmitted": 1, "Not Readmitted": 0}).to_numpy()
            fig = plot_calibration_curve(labels, df["Readmission Prob."], show_plot=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("True outcome labels or predicted probabilities not available to plot calibration curve.")
