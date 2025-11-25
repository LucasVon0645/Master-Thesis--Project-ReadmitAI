from typing import Optional

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve

from app.utils import format_feature_value

def plot_probability_distribution(df: pd.DataFrame, prob_threshold: float) -> go.Figure:
    if "True Outcome" in df.columns:
        color_col = "True Outcome"
        title = "Distribution of Readmission Probabilities by True Outcome"
    else:
        color_col = None
        title = "Distribution of Readmission Probabilities"

    fig = px.histogram(
        df,
        x="Readmission Prob.",
        nbins=50,
        color=color_col,
        title=title,
        labels={"Readmission Prob.": "Predicted Readmission Probability"},
        height=400,
    )

    fig.update_layout(
        bargap=0.1,
        yaxis_title="Count",
        legend=dict(font=dict(size=20)),
        title=dict(font=dict(size=24)),
    )
    fig.update_xaxes(title_font=dict(size=20))
    fig.update_yaxes(title_font=dict(size=20))

    fig.add_vline(
        x=prob_threshold,
        line_dash="dash",
        line_color="red",
        line_width=3,   # makes the line thicker
        annotation_position="top right",
        annotation=dict(
            text="Threshold",
            font=dict(size=20, color="red"),
            yshift=15,   # move annotation upward
        ),
    )
    return fig

def plot_subject_evolution(df, subject_id,
                           features_to_plot: Optional[list] = None,
                           save_html_file_path: Optional[str] = None,
                           textposition='outside', extend_time_horizon_by=365, show: bool = False):
    # Filter for the patient
    patient_df = df[df['SUBJECT_ID'] == subject_id].copy()
    patient_df['ADMITTIME'] = pd.to_datetime(patient_df['ADMITTIME'])
    patient_df['DISCHTIME'] = pd.to_datetime(patient_df['DISCHTIME'])
    patient_df = patient_df.sort_values('ADMITTIME')

    # Features to track over time (excluding ID/time columns)
    if features_to_plot is None:
        # Default features to plot if not provided
        features_to_plot = [
            'HOSPITALIZATION_DAYS', 'NUM_COMORBIDITIES',
            'NUM_PREV_HOSPITALIZATIONS', 'DAYS_SINCE_LAST_HOSPITALIZATION',
            'DAYS_IN_ICU', 'NUM_DRUGS', 'NUM_PROCEDURES'
        ]

    # Melt the data so each feature is a row
    melted = patient_df.melt(
        id_vars=['ADMITTIME', 'DISCHTIME', 'HADM_ID'],
        value_vars=features_to_plot,
        var_name='Feature',
        value_name='Value'
    )

    # Convert all values to string for display
    melted['Value'] = melted['Value'].apply(format_feature_value)

    # Create the plot
    fig = px.timeline(
        melted,
        x_start='ADMITTIME',
        x_end='DISCHTIME',
        y='Feature',
        text='Value',
        title=f'Evolution of SUBJECT_ID {subject_id}',
    )
    fig.update_traces(textposition=textposition, textfont_size=10)
    # Adjust layout to avoid text being cut off
    fig.update_layout(
        height=600,
        yaxis_title="Feature",
        xaxis_title="Time",
        margin=dict(l=100, r=100, t=50, b=50),
        xaxis=dict(
            range=[
                melted["ADMITTIME"].min() - pd.Timedelta(days=100),
                melted["DISCHTIME"].max() + pd.Timedelta(days=extend_time_horizon_by),
            ]
        ),
    )

    if save_html_file_path:
        if not save_html_file_path.endswith('.html'):
            save_html_file_path += '.html'
        # Save the plot as a html image
        fig.write_html(save_html_file_path)
        print(f"Plot saved as {save_html_file_path}")

    if show:
        fig.show()

    return fig

def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: list[str],
):
    fig = go.Figure(
        data=go.Heatmap(
            z=conf_matrix,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            text=conf_matrix,
            texttemplate="%{text}",
            textfont=dict(size=24),
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
            colorbar=dict(
                title="Count"
            ),
        )
    )

    fig.update_layout(
        title=dict(text="Confusion Matrix", font=dict(size=24)),
        xaxis=dict(
            title="Predicted label",
            tickfont=dict(size=16),
        ),
        yaxis=dict(
            title="True label",
            tickfont=dict(size=16),
            autorange="reversed",
        ),
        template="plotly_white",
        width=600,
        height=600,
    )
    
    fig.update_xaxes(title_font=dict(size=18))
    fig.update_yaxes(title_font=dict(size=18))

    return fig

def plot_calibration_curve(labels, pred_prob, n_bins=5, title="Calibration Curve",
                           save_path: Optional[str] = None, show_plot=True):

    fraction_of_positives, mean_predicted_value = calibration_curve(
        labels, pred_prob, n_bins=n_bins
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=mean_predicted_value,
        y=fraction_of_positives,
        mode='lines+markers',
        name='Calibration Curve'
    ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfectly Calibrated',
        line=dict(dash='dash')
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=24)),
        xaxis=dict(
            title="Mean Predicted Probability",
            tickfont=dict(size=16),
            range=[0, 1],
        ),
        yaxis=dict(
            title="Fraction of Positives",
            tickfont=dict(size=16),
            range=[0, 1],
        ),
        legend=dict(
            title=dict(text="Legend", font=dict(size=18)),
            font=dict(size=16)
        ),
        template="plotly_white",
        width=800,
        height=400
    )
    
    fig.update_xaxes(title_font=dict(size=18))
    fig.update_yaxes(title_font=dict(size=18))

    if save_path:
        if not save_path.endswith('.html'):
            save_path += '.html'
        fig.write_html(save_path)

    if show_plot:
        fig.show()

    return fig

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

def plot_feature_attributions(
    feat_attr_df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Feature Attributions",
    feature_col: str = "feature",
    attr_col: str = "attribution",
    top_k: Optional[int] = None,
):
    """
    Plot feature attributions as a horizontal bar chart and save to output_path.
    """
    
    # Sort features by absolute attribution value
    feat_attr_df = feat_attr_df.sort_values(by=attr_col, key=lambda x: x.abs(), ascending=False)
    
    if top_k is not None:
        feat_attr_df = feat_attr_df.head(top_k)

    fig = px.bar(
        feat_attr_df,
        x=attr_col,
        y=feature_col,
        title=title,
        orientation="h",
    )
    if output_path:
        fig.write_html(output_path)
    
    return fig
