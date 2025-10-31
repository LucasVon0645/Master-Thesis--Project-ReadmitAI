from __future__ import annotations
import pandas as pd
from typing import Any, Dict, List
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

def read_csv_to_dataframe(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df

def build_predictions_dataframe(predictions: List[Dict[str, Any]]) -> pd.DataFrame:
    if not predictions:
        return pd.DataFrame(columns=["SUBJECT_ID", "HADM_ID", "Readmission Probability"])
    else:
        data_dict = {
                "SUBJECT_ID": predictions["subject_ids"],
                "HADM_ID": predictions["hadm_ids"],
                "Readmission Prob.": predictions["pred_probs"],
            }
        if predictions.get("true_labels") is not None:
            true_labels: list = predictions["true_labels"]

            data_dict["True Outcome"] = list(map(lambda x: "Readmitted" if x == 1 else "Not Readmitted", true_labels))
        return pd.DataFrame(data_dict)

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