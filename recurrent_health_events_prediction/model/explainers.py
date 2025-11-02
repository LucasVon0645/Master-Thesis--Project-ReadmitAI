from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch

from recurrent_health_events_prediction.model.NextEventPredictionModel import NextEventPredictionModel
from recurrent_health_events_prediction.model.model_types import SurvivalModelType

def make_predict_surv_prob_at_t(model: NextEventPredictionModel, t: float):
    def predict_surv_prob_at_t(X_input):
        if not isinstance(X_input, pd.DataFrame):
            X_input = pd.DataFrame(X_input, columns=model.feature_names_in_)
        
        surv_df = model.predict_survival(X_input, times=[t])
        return surv_df.loc[t].values  # 1D array of survival probs at time t
    return predict_surv_prob_at_t

def explain_survival_model_prob(model: NextEventPredictionModel, X_train, X_explain, t: float):
    predict_fn = make_predict_surv_prob_at_t(model, t)
    if model.model_type == SurvivalModelType.GBM:
        # For GBM survival models, we need to ensure the input is in the right format
        explainer = shap.TreeExplainer(predict_fn, X_train)
    else:
        explainer = shap.KernelExplainer(predict_fn, X_train)

    shap_values = explainer.shap_values(X_explain)

    return shap_values, explainer

def plot_waterfall(explainer, X_explain, shap_values: np.ndarray, feature_names=None, max_display=10):
    """
    Plot a SHAP waterfall plot for the given SHAP values.
    
    Args:
        shap_values: SHAP values to plot of a single instance.
        feature_names: Optional list of feature names.
        max_display: Maximum number of features to display.
        
    Returns:
        A plotly figure showing the SHAP waterfall plot.
    """

    if len(X_explain) != 1:
        raise ValueError("X_explain must contain exactly one row for waterfall plot.")
    if shap_values.ndim != 1:
        raise ValueError("shap_values must be a 1D array for waterfall plot.")

    shap.plots.waterfall(shap.Explanation(values=shap_values,
                                      base_values=explainer.expected_value,
                                      data=X_explain.iloc[0],
                                      feature_names=feature_names),
                                      max_display=max_display)

def plot_survival_shap_summary(model: NextEventPredictionModel, X_train, X_explain, t=30, title=None):
    """
    Generate and return a SHAP summary plot figure for survival probability at time `t`.
    
    Parameters:
    - model: trained NextEventPredictionModel
    - X_train: background data (usually training data)
    - X_explain: data to explain (e.g., test data or subset of training)
    - t: float, time at which to compute survival probabilities
    - title: optional plot title
    
    Returns:
    - fig: matplotlib.figure.Figure
    """
    feature_names = model.feature_names_in_
    # Ensure only selected features are used
    X_train_subset = X_train[feature_names].copy()
    X_explain_subset = X_explain[feature_names].copy()
    
    # Get SHAP values
    shap_values, _ = explain_survival_model_prob(
        model, X_train_subset, X_explain_subset, t=t
    )
    
    # Handle binary/multiclass shape
    if shap_values.ndim == 3:
        shap_values_to_plot = shap_values[:, :, 1]  # Class 1 survival
    else:
        shap_values_to_plot = shap_values

    # Create figure for SHAP plot
    fig = plt.figure()
    plt.title(title or f"SHAP Summary Plot (t={t})", pad=20)
    shap.summary_plot(shap_values_to_plot, 
                      X_explain_subset.astype(float),
                      feature_names=feature_names,
                      show=False)  # Don't auto-show so we can return fig
    return fig


# Deep Learning model explainability (Integrated Gradients)
from captum.attr import IntegratedGradients, LayerIntegratedGradients

@torch.no_grad()
def compute_train_stats(train_loader, max_rows_for_median: int = 200_000):
    """
    Compute mean and median of features in training data.
    It considers only valid (non-masked) entries for the past features.
    Returns:
      {
        "mean_curr":   [D_curr],
        "mean_past":   [D_long],
        "median_curr": [D_curr]  (if collected),
        "median_past": [D_long]  (if collected),
        "has_median":  bool
      }
    """
    sum_curr = None
    sum_past = None
    n_curr = 0
    n_past = 0

    # buffers for median (on CPU)
    buf_curr = []
    buf_past = []
    total_rows_curr = 0
    total_rows_past = 0
    collect_median = True  # we will try; if it exceeds the limit, we fall back

    for batch in train_loader:
        # Support for datasets that return (x_curr, x_past, mask, y) or (x_curr, x_past, mask)
        if len(batch) == 4:
            x_curr, x_past, mask, _ = batch
        else:
            x_curr, x_past, mask = batch

        x_curr = x_curr.to("cpu", non_blocking=True)
        x_past = x_past.to("cpu", non_blocking=True)
        mask = mask.to("cpu", non_blocking=True).bool()

        B, T, D_long = x_past.shape
        D_curr = x_curr.shape[-1]

        # --- means ---
        # current
        sum_curr = (sum_curr if sum_curr is not None else torch.zeros(D_curr)) + x_curr.sum(dim=0)
        n_curr += x_curr.size(0)

        # past: use only valid rows
        valid_rows = x_past[mask]        # [N_valid, D_long]
        if valid_rows.numel() > 0:
            sum_past = (sum_past if sum_past is not None else torch.zeros(D_long)) + valid_rows.sum(dim=0)
            n_past += valid_rows.shape[0]

        # --- medianas (opcional, até limite de linhas) ---
        if collect_median:
            # current
            if total_rows_curr + x_curr.size(0) <= max_rows_for_median:
                buf_curr.append(x_curr)
                total_rows_curr += x_curr.size(0)
            else:
                collect_median = False
            # past
            if valid_rows.numel() > 0:
                if total_rows_past + valid_rows.shape[0] <= max_rows_for_median:
                    buf_past.append(valid_rows)
                    total_rows_past += valid_rows.shape[0]
                else:
                    collect_median = False

    mean_curr = sum_curr / max(n_curr, 1)
    mean_past = sum_past / max(n_past, 1)

    stats = {
        "mean_curr": mean_curr,
        "mean_past": mean_past,
        "has_median": False,
    }

    if collect_median and len(buf_curr) > 0 and len(buf_past) > 0:
        curr_mat = torch.cat(buf_curr, dim=0)          # [N_curr, D_curr]
        past_mat = torch.cat(buf_past, dim=0)          # [N_past, D_long]
        median_curr = torch.quantile(curr_mat, 0.5, dim=0)   # [D_curr]
        median_past = torch.quantile(past_mat, 0.5, dim=0)   # [D_long]
        stats.update({
            "median_curr": median_curr,
            "median_past": median_past,
            "has_median": True
        })

    return stats

def _make_baselines(x_curr, x_past, mask, strategy="zeros", stats=None):
    """
    Create baseline tensors for current and past features according to strategy.
    Args:
      x_curr: [B, D_curr] tensor of current features
      x_past: [B, T, D_long] tensor of past features
      mask:   [B, T] boolean tensor indicating valid past steps
      strategy: "zeros", "means", or "medians"
      stats: precomputed statistics dict from compute_train_stats (required for "means" or "medians")
    Returns:
      base_curr: [B, D_curr] baseline tensor for current features
      base_past: [B, T, D_long] baseline tensor for past features
    """
    if strategy == "zeros":
        base_curr = torch.zeros_like(x_curr)
        base_past = torch.zeros_like(x_past)

    elif strategy == "means":
        assert stats is not None and "mean_curr" in stats and "mean_past" in stats, \
            "Passe stats com mean_curr/mean_past para strategy='means'."
        base_curr = stats["mean_curr"].to(x_curr).expand_as(x_curr).clone()
        base_past = stats["mean_past"].to(x_past).expand_as(x_past).clone()

    elif strategy == "medians":
        assert stats is not None and stats.get("has_median", False), \
            "Medianas não disponíveis (a coleta pode ter sido desativada por limite)."
        base_curr = stats["median_curr"].to(x_curr).expand_as(x_curr).clone()
        base_past = stats["median_past"].to(x_past).expand_as(x_past).clone()

    else:
        raise ValueError(f"Estratégia de baseline desconhecida: {strategy}")

    # Zerar baseline nos passos preenchidos (padding) segundo a máscara
    base_past = base_past.masked_fill(~mask.unsqueeze(-1), 0.0)
    return base_curr, base_past

def _forward_for_attr(model, x_curr, x_past, mask):
    out = model(x_current=x_curr, x_past=x_past, mask_past=mask)
    if isinstance(out, tuple):
        logits, _ = out
    else:
        logits = out
    return logits

def _explain_batch(
    model: torch.nn.Module,
    x_curr: torch.Tensor,       # [B, D_curr]
    x_past: torch.Tensor,       # [B, T, D_long]
    mask: torch.Tensor,         # [B, T] (bool) or [B, T, 1]
    *,
    baseline_strategy: str = "means",   # "zeros" | "means" | "medians"
    stats: Optional[Dict[str, torch.Tensor]] = None,  # required for means/medians
    n_steps: int = 64,
    internal_batch_size: int = 64,
    layer_for_split: Optional[torch.nn.Module] = None # e.g. model.classifier_head[0] to split [history||current]
) -> Dict[str, torch.Tensor]:
    """
    Returns:
      {
        'attr_curr':              [B, D_curr] (signed IG attributions),
        'attr_past':              [B, T, D_long] (signed),
        'curr_importance':        [B, D_curr] (abs),
        'past_feat_importance':   [B, D_long] (abs, summed over time),
        'time_importance':        [B, T] (abs, summed over features),
        # optional if layer_for_split is provided:
        'hist_vs_curr_history':   [B],  (abs contrib at fc1 input, history summary part)
        'hist_vs_curr_current':   [B],  (abs contrib at fc1 input, current vector part)
      }
    """
    model.eval()

    # --- devices & shapes ---
    device = next(model.parameters()).device
    x_curr = x_curr.to(device).requires_grad_(True)
    x_past = x_past.to(device).requires_grad_(True)
    mask = mask.to(device)
    if mask.dim() == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    mask = mask.bool()

    # --- baselines per batch ---
    base_curr, base_past = _make_baselines(
        x_curr, x_past, mask,
        strategy=baseline_strategy,
        stats=stats
    )

    # --- IG setup ---
    ig = IntegratedGradients(lambda x_c, x_p, m: _forward_for_attr(model, x_c, x_p, m))

    any_past_obs = mask.any().item()
    
    if not any_past_obs:
            attr_past = None
            attr_curr = ig.attribute(
                inputs=x_curr,
                baselines=(base_curr,),
                target=None,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,
                additional_forward_args=(x_past, mask,),
            )
    else:
        # run IG
        attr_curr, attr_past = ig.attribute(
            inputs=(x_curr, x_past),
            baselines=(base_curr, base_past),
            target=None,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            additional_forward_args=(mask,),
        )

    # --- per-sample aggregations (absolute magnitude = importance) ---
    # current features: [B, D_curr]
    curr_importance = attr_curr.abs()

    if any_past_obs:
        # past features aggregated over time: [B, D_long]
        past_feat_importance = attr_past.abs().sum(dim=1)
        # time importance aggregated over features: [B, T]
        time_importance = attr_past.abs().sum(dim=2)
    else:
        past_feat_importance = torch.zeros((x_past.size(0), x_past.size(2)), device=device)
        time_importance = torch.zeros((x_past.size(0), x_past.size(1)), device=device)

    out = {
        "attr_curr": attr_curr.detach().cpu(),
        "attr_past": attr_past.detach().cpu() if attr_past is not None else None,
        "curr_importance": curr_importance.detach().cpu(),
        "past_feat_importance": past_feat_importance.detach().cpu(),
        "time_importance": time_importance.detach().cpu(),
    }
    
    if any_past_obs and layer_for_split is not None:
    # --- optional: split [history||current] at input of fc1 (LayerIntegratedGradients) ---
        lig = LayerIntegratedGradients(lambda x_c, x_p, mask: _forward_for_attr(model, x_c, x_p, mask),
                                    layer_for_split)
        
        lig_attr = lig.attribute(
            inputs=(x_curr, x_past),
            baselines=(base_curr, base_past),
            target=None,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            additional_forward_args=(mask,)
        )  # [B, H_seq + D_curr] at the layer input

        H_seq = getattr(model, "hidden_size_seq", None)
        if H_seq is None:
            # fallback: try to infer from layer input size minus D_curr
            H_seq = lig_attr.shape[1] - x_curr.shape[1]
        lig_abs = lig_attr.abs()
        out["hist_vs_curr_history"] = lig_abs[:, :H_seq].sum(dim=1).detach().cpu()
        out["hist_vs_curr_current"] = lig_abs[:, H_seq:].sum(dim=1).detach().cpu()

    return out

def explain_deep_learning_model_feat(
    model,
    x_curr,
    x_past,
    mask,
    feature_names_curr,
    feature_names_past,
    layer_for_split=None,
    baseline_strategy="means",
    stats=None,
    n_steps=64,
    internal_batch_size=16,
):
    """
    Run explain_batch on a batch or single sample and return DataFrames for
    current and past feature importances.

    Parameters
    ----------
    model : nn.Module
    x_curr, x_past, mask : torch.Tensor
        Model inputs (single sample or batch). If single sample, automatically
        adds batch dimension.
    feature_names_curr, feature_names_past : list[str]
        Feature names aligned with tensor dimensions.
    explain_fn : callable
        Function like `explain_batch(model, x_curr, x_past, mask, ...)`.
    layer_for_split : torch.nn.Module, optional
        Layer used for split explanation (as in your example).
    Returns
    -------
    df_curr_all : pd.DataFrame
        Columns: [sample_idx, feature_name, importance]
    df_past_all : pd.DataFrame
        Columns: [sample_idx, feature_name, importance]
    df_split : pd.DataFrame
        Columns: [sample_idx, past_importance, current_importance]
    """
    # --- handle single-sample case ---
    if x_curr.ndim == 1:
        x_curr = x_curr.unsqueeze(0)
    if x_past.ndim == 2:
        x_past = x_past.unsqueeze(0)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)

    # --- run explanation ---
    res = _explain_batch(
        model,
        x_curr,
        x_past,
        mask,
        baseline_strategy=baseline_strategy,
        stats=stats,
        n_steps=n_steps,
        layer_for_split=layer_for_split,
        internal_batch_size=internal_batch_size,
    )

    # --- build DataFrames ---
    df_curr_list = []
    df_past_list = []

    for i in range(len(res["curr_importance"])):
        # Current features
        vals_curr = res["curr_importance"][i].detach().cpu().numpy()
        df_curr = pd.DataFrame({
            "sample_idx": i,
            "feature": feature_names_curr,
            "attribution": vals_curr
        })
        df_curr_list.append(df_curr)

        # Past features
        vals_past = res["past_feat_importance"][i].detach().cpu().numpy()
        df_past = pd.DataFrame({
            "sample_idx": i,
            "feature": feature_names_past,
            "attribution": vals_past
        })
        df_past_list.append(df_past)

    df_curr_all = pd.concat(df_curr_list, ignore_index=True)
    df_past_all = pd.concat(df_past_list, ignore_index=True)
    
    if "hist_vs_curr_history" in res and "hist_vs_curr_current" in res:
        df_split = pd.DataFrame({
            "sample_idx": list(range(len(res["hist_vs_curr_history"]))),
            "past_attribution": res["hist_vs_curr_history"].numpy(),
            "current_attribution": res["hist_vs_curr_current"].numpy(),
        })

    return df_curr_all, df_past_all, df_split