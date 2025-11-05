import copy
import os
from typing import Optional
import numpy as np
import optuna
from pathlib import Path
from optuna.importance import get_param_importances
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yaml

from recurrent_health_events_prediction.utils.neptune_utils import add_plotly_plots_to_neptune_run, upload_file_to_neptune

# =========================
#  Hyperparameter Tuning Utils
# =========================

def save_space_to_txt(space_hyperparams, out_dir):
    """
    Save your hyperparameter search space (dict of lists/tuples)
    to a plain text file.
    """
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, f"hyperparams_search_space.txt")

    with open(filepath, "w") as f:
        f.write("### Hyperparameter Search Space ###\n\n")
        for name, values in space_hyperparams.items():
            # pretty formatting for tuples/lists/numbers
            if isinstance(values, (list, tuple)):
                f.write(f"{name}: {list(values)}\n")
            else:
                f.write(f"{name}: {values}\n")

    print(f"Search space saved to: {filepath}")
    return filepath


def save_study_artifacts(
    study: optuna.study.Study,
    out_dir: str,
    base_config: dict,
    model_class_name: str,
    neptune_run=None
):
    """
    Writes:
      - best_params.yaml
      - trials.csv
      - param_importances.yaml
      - (optional) best_config.yaml and best_model.pth after recomputing metrics
    Also prints best metrics captured during HPO (from trial.user_attrs).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- a) Best params → YAML
    best_params = study.best_trial.params
    best_params_path = out / "best_params.yaml"
    print(f"Best params saved to: {best_params_path}")
    with open(best_params_path, "w") as f:
        yaml.safe_dump(best_params, f, sort_keys=True)
    if neptune_run:
        neptune_run["best_params"] = best_params

    # --- b) Trials → CSV
    df = study.trials_dataframe(attrs=(
        "number", "value", "params", "state", "datetime_start", "datetime_complete"
    ))
    trials_path = out / "trials.csv"
    print(f"Trials saved to: {trials_path}")
    df.to_csv(trials_path, index=False)
    if neptune_run:
        upload_file_to_neptune(
            neptune_run,
            local_path=str(trials_path),
            neptune_base_path="artifacts",
            neptune_filename="trials.csv"
        )

    # --- c) Parameter importances → YAML (may fail if too few trials)
    try:
        imps = get_param_importances(study)
        if neptune_run:
            neptune_run["param_importances"] = imps
        with open(out / "param_importances.yaml", "w") as f:
            yaml.safe_dump({k: float(v) for k, v in imps.items()}, f, sort_keys=False)
    except Exception as e:
        print(f"[warn] Could not compute importances: {e}")

    # --- d) Best metrics as recorded during tuning (no retrain)
    bt = study.best_trial
    best_auroc = bt.user_attrs.get("auroc")
    best_f1    = bt.user_attrs.get("f1")
    print(f"Best trial number: {bt.number}")
    print(f"Best recorded metrics (during HPO): AUROC={best_auroc}, F1={best_f1}")
    
    if neptune_run:
        neptune_run["best_trial_number"] = bt.number
        neptune_run["best_auroc"] = best_auroc
        neptune_run["best_f1"] = best_f1
    
    # Build best config for reference
    best_config = copy.deepcopy(base_config)

    # Inject best hyperparams back into config
    # (must match how you injected inside objective())
    p = best_params
    best_config["learning_rate"] = p["learning_rate"]
    best_config["batch_size"] = p["batch_size"]
    best_config["weight_decay"] = p["weight_decay"]
    best_config["lr_scheduler"] = p["lr_scheduler"]
    best_config["model_params"]["hidden_size_head"] = p["hidden_size_head"]
    best_config["model_params"]["dropout"] = p["dropout"]
    best_config["model_params"]["hidden_size_seq"] = p["hidden_size_seq"]

    if model_class_name == "GRUNet":
        best_config["model_class"] = "GRUNet"
        best_config["model_params"]["num_layers_seq"]  = p["num_layers_seq"]
    elif model_class_name == "CrossAttnPoolingNet":
        best_config["model_class"] = "CrossAttnPoolingNet"
        best_config["model_params"]["num_heads"] = p["num_heads"]
        best_config["model_params"]["use_posenc"] = p["use_posenc"]
    elif model_class_name == "AttentionPoolingNet":
        best_config["model_class"] = "AttentionPoolingNet"
        # no extra params
    elif model_class_name == "AttentionPoolingNetCurrentQuery":
        best_config["model_class"] = "AttentionPoolingNetCurrentQuery"
        best_config["model_params"]["use_separate_values"] = p["use_separate_values"]
        best_config["model_params"]["scale_scores"] = p["scale_scores"]

    # Save best config
    best_config_path = out / "best_config.yaml"
    with open(best_config_path, "w") as f:
        yaml.safe_dump(best_config, f, sort_keys=False)
    print(f"Best config saved to: {best_config_path}")
    if neptune_run:
        upload_file_to_neptune(
            neptune_run,
            local_path=str(best_config_path),
            neptune_base_path="artifacts",
            neptune_filename="best_config.yaml"
        )

# =========================
#  Feature Selection Utils
# =========================

def rank_features(attr_df, score_col="attribution_activity"):
    """Return a DataFrame sorted by mean absolute attribution per feature."""
    agg = (
        attr_df.groupby("feature", as_index=False)[score_col]
        .agg(lambda x: np.mean(np.abs(x)))
    )
    return agg.sort_values(score_col, ascending=False, ignore_index=True)


def select_topk_features(curr_attr_df, past_attr_df,
                         current_feat_cols, longitudinal_feat_cols,
                         k_current=None, k_past=None,
                         p_current=None, p_past=None):
    """Return top-k (or top-% if p_) features within each group."""
    curr_rank = rank_features(curr_attr_df)
    past_rank = rank_features(past_attr_df)

    kc = k_current or max(1, int(round(p_current / 100 * len(current_feat_cols))))
    kp = k_past or max(1, int(round(p_past / 100 * len(longitudinal_feat_cols))))

    keep_curr = set(curr_rank.head(kc)["feature"])
    keep_past = set(past_rank.head(kp)["feature"])

    new_curr = [f for f in current_feat_cols if f in keep_curr]
    new_past = [f for f in longitudinal_feat_cols if f in keep_past]
    return new_curr, new_past


def make_auc_vs_features_plot(results, title="AUC vs Total Features"):
    """
    Create a Plotly line plot showing AUROC vs total number of features.
    
    Parameters
    ----------
    results : list[dict]
        Output list from run_grouped_feature_selection().
    title : str
        Title of the plot.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure.
    """
    df = pd.DataFrame(results).copy()
    df["total_features"] = df.get("n_current", 0) + df.get("n_past", 0)
    df = df.sort_values("total_features")

    # Build custom hover text
    hover_text = (
        "Total features: %{x}<br>"
        "AUROC: %{y:.4f}<br>"
        "k_current: %{customdata[0]}<br>"
        "k_past: %{customdata[1]}"
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["total_features"],
        y=df["AUROC"],
        mode="lines+markers",
        name="AUC",
        line=dict(width=2),
        marker=dict(size=8),
        customdata=df[["k_current", "k_past"]].values,
        hovertemplate=hover_text
    ))

    # Highlight baseline if present
    if "all" in df["k_current"].astype(str).values:
        baseline = df[df["k_current"].astype(str) == "all"].iloc[0]
        fig.add_trace(go.Scatter(
            x=[baseline["total_features"]],
            y=[baseline["AUROC"]],
            mode="markers+text",
            text=["Baseline"],
            textposition="top center",
            marker=dict(color="black", size=10, symbol="x"),
            name="Baseline",
            hovertemplate=(
                f"Baseline<br>"
                f"Total features: {baseline['total_features']}<br>"
                f"AUROC: {baseline['AUROC']:.4f}"
            )
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Total number of features",
        yaxis_title="AUROC",
        template="plotly_white",
        hovermode="x unified",
        width=700,
        height=450,
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

def make_feature_attr_plots(curr_attr_df, past_attr_df, neptune_run):
    # Plot attributions
    fig_curr = plot_feature_attributions(
        curr_attr_df,
        title="Current Features - IG Attributions (Importance)",
        feature_col="feature",
        attr_col="attribution_activity",
    )
    fig_past = plot_feature_attributions(
        past_attr_df,
        title="Longitudinal Features - IG Attributions (Importance)",
        feature_col="feature",
        attr_col="attribution_activity",
    )

    fig_direction_curr = plot_feature_attributions(
        curr_attr_df,
        title="Current Features - IG Attributions (Effect-Direction)",
        feature_col="feature",
        attr_col="attribution_direction",
    )

    fig_direction_past = plot_feature_attributions(
        past_attr_df,
        title="Longitudinal Features - IG Attributions (Effect-Direction)",
        feature_col="feature",
        attr_col="attribution_direction",
    )

    add_plotly_plots_to_neptune_run(
        neptune_run, fig_curr, "feature_importance_curr", "plots"
    )
    add_plotly_plots_to_neptune_run(
        neptune_run, fig_past, "feature_importance_past", "plots"
    )
    add_plotly_plots_to_neptune_run(
        neptune_run, fig_direction_curr, "feature_direction_curr", "plots"
    )
    add_plotly_plots_to_neptune_run(
        neptune_run, fig_direction_past, "feature_direction_past", "plots"
    )