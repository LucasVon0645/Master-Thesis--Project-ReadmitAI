import copy

import os
from importlib import resources as impresources

import torch
import yaml

from recurrent_health_events_prediction import configs
from recurrent_health_events_prediction.model.explainers import (
    global_feature_importance,
)
from recurrent_health_events_prediction.model_selection.deep_learning.utils import (
    make_auc_vs_features_plot,
    make_feature_attr_plots,
    select_topk_features,
)
from recurrent_health_events_prediction.training.train_deep_learning_models import (
    evaluate,
    train,
)
from recurrent_health_events_prediction.training.utils_deep_learning import (
    build_or_load_datasets,
)
from recurrent_health_events_prediction.utils.general_utils import check_if_file_exists, import_yaml_config
from recurrent_health_events_prediction.utils.neptune_utils import add_plotly_plots_to_neptune_run, initialize_neptune_run


# =========================
#  Core experiment function
# =========================

def run_grouped_feature_selection(
    model_config,
    training_data_config,
    preproc_train_csv,
    preproc_eval_csv,
    cache_pytorch_datasets_path,
    train_pt_name,
    eval_pt_name,
    overwrite_pt,
    k_sweep=((5, 5), (10, 10), (20, 20), (50, 50)),
    neptune_run=None,
):
    """Run baseline + grouped top-k feature selection, logging to Neptune if enabled."""

    # ----- Baseline: all features -----
    print("\n=== Baseline: training with all features ===")
    base_cfg = copy.deepcopy(model_config)
    train_ds, val_ds, _ = build_or_load_datasets(
        preproc_train_csv=preproc_train_csv,
        preproc_eval_csv=preproc_eval_csv,
        model_config=base_cfg,
        training_data_config=training_data_config,
        cache_dir=cache_pytorch_datasets_path,
        train_pt_name=f"{train_pt_name}_baseline",
        eval_pt_name=f"{eval_pt_name}_baseline",
        overwrite_pt=overwrite_pt,
    )

    model, loss_epochs = train(
        train_ds,
        base_cfg,
        neptune_run=neptune_run,
        writer=None,
        val_dataset=val_ds,
        early_stopping_patience=5,
    )

    results, _, _, _, _ = evaluate(
        val_ds, model, batch_size=base_cfg["batch_size"]
    )

    auroc_base = results["auroc"]
    f1_base = results["f1_score"]

    print(f"Baseline AUROC: {auroc_base:.4f}, F1: {f1_base:.4f}")

    if neptune_run:
        neptune_run["baseline/auroc"] = auroc_base
        neptune_run["baseline/f1_score"] = f1_base
        neptune_run["baseline/n_current"] = len(base_cfg["current_feat_cols"])
        neptune_run["baseline/n_past"] = len(base_cfg["longitudinal_feat_cols"])
        neptune_run["baseline/features/current"] = base_cfg["current_feat_cols"]
        neptune_run["baseline/features/past"] = base_cfg["longitudinal_feat_cols"]

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=model_config["batch_size"], shuffle=True
    )

    # ----- Compute attributions on baseline model -----
    print("\nComputing feature attributions on baseline model...")
    curr_attr_df, past_attr_df, mean_abs_time = global_feature_importance(
        model=model,
        train_loader=train_loader,
        test_loader=train_loader,
        current_feat_cols=base_cfg["current_feat_cols"],
        longitudinal_feat_cols=base_cfg["longitudinal_feat_cols"],
    )

    if neptune_run:
        make_feature_attr_plots(curr_attr_df, past_attr_df, mean_abs_time, neptune_run)

    n_curr = len(base_cfg["current_feat_cols"])
    n_past = len(base_cfg["longitudinal_feat_cols"])
    # ----- Run sweep -----
    all_results = [{"k_current": n_curr, "k_past": n_past, "AUROC": auroc_base, "F1": f1_base, "n_current": n_curr, "n_past": n_past}]

    for (k_curr, k_past) in k_sweep:
        print(f"\n=== Running top-k sweep: current={k_curr}, past={k_past} ===")

        # 1. Select top features per group
        new_curr, new_past = select_topk_features(
            curr_attr_df, past_attr_df,
            base_cfg["current_feat_cols"], base_cfg["longitudinal_feat_cols"],
            k_current=k_curr, k_past=k_past
        )

        # 2. Update model config
        cfg = copy.deepcopy(base_cfg)
        cfg["current_feat_cols"] = new_curr
        cfg["longitudinal_feat_cols"] = new_past
        cfg["model_params"]["input_size_curr"] = len(new_curr)
        cfg["model_params"]["input_size_seq"] = len(new_past)

        # 3. Build datasets
        train_ds, val_ds, _ = build_or_load_datasets(
            preproc_train_csv=preproc_train_csv,
            preproc_eval_csv=preproc_eval_csv,
            model_config=cfg,
            training_data_config=training_data_config,
            cache_dir=cache_pytorch_datasets_path,
            train_pt_name=f"{train_pt_name}_kcurr{k_curr}_kpast{k_past}",
            eval_pt_name=f"{eval_pt_name}_kcurr{k_curr}_kpast{k_past}",
            overwrite_pt=overwrite_pt,
        )

        # 4. Train model
        model, loss_epochs = train(
            train_ds,
            cfg,
            neptune_run=neptune_run,
            writer=None,
            val_dataset=val_ds,
            early_stopping_patience=5,
        )

        num_epochs = len(loss_epochs)
        loss = loss_epochs[-1] if loss_epochs else None

        # 5. Evaluate model
        results, _, _, _, _ = evaluate(
            val_ds, model, batch_size=cfg["batch_size"]
        )

        auroc = results["auroc"]
        f1 = results["f1_score"]
        n_curr = len(new_curr)
        n_past = len(new_past)

        print(f"AUROC: {auroc:.4f}, F1: {f1:.4f}")

        # 6. Log to Neptune
        if neptune_run:
            run_path = f"current_{k_curr}_past_{k_past}"
            neptune_run[f"{run_path}/auroc"] = auroc
            neptune_run[f"{run_path}/f1_score"] = f1
            neptune_run[f"{run_path}/n_current"] = n_curr
            neptune_run[f"{run_path}/n_past"] = n_past
            neptune_run[f"{run_path}/features/current"] = new_curr
            neptune_run[f"{run_path}/features/past"] = new_past
            neptune_run[f"{run_path}/num_epochs"] = num_epochs
            neptune_run[f"{run_path}/loss"] = loss

        # 7. Collect results locally
        all_results.append({
            "k_current": k_curr,
            "k_past": k_past,
            "AUROC": auroc,
            "F1": f1,
            "n_current": n_curr,
            "n_past": n_past,
        })

    fig = make_auc_vs_features_plot(all_results)
    if neptune_run:
        add_plotly_plots_to_neptune_run(neptune_run, fig, "auc_vs_features", "plots")
    fig.write_html("auc_vs_features.html")

    best = max(all_results, key=lambda r: r["AUROC"])
    print(f"Best config: {best}")
    
    best_new_curr, best_new_past = select_topk_features(
        curr_attr_df, past_attr_df,
        base_cfg["current_feat_cols"], base_cfg["longitudinal_feat_cols"],
        k_current=best["k_current"], k_past=best["k_past"]
    )

    # ----- Finish -----
    if neptune_run:
        neptune_run["best/summary"] = best
        neptune_run["best/current_features"] = '\n'.join(best_new_curr)
        neptune_run["best/past_features"] = '\n'.join(best_new_past)
        neptune_run.stop()
        print(f"\nBest config logged to Neptune: {best}")

    return all_results, best

def main(
    neptune_run_name: str,
    model_config_path: str,
    k_sweep: tuple,
    training_data_config: dict,
    preproc_train_csv: str,
    preproc_eval_csv: str,
    cache_pytorch_datasets_path: str,
    log_in_neptune: bool = True,
    multiple_hosp_patients=True):

    model_config = import_yaml_config(model_config_path)
    model_config_dir_path = os.path.dirname(model_config_path)
    model_name = model_config.get("model_name", "unknown_model")
    model_dir_name = os.path.basename(model_config_dir_path)

    print(f"Using model config from: {model_config_path}")
    print(f"Model name: {model_name}")

    neptune_run = None
    if log_in_neptune:
        neptune_tags = ["feature_selection", "captum"]
        neptune_tags.append("multiple_hosp_patients" if multiple_hosp_patients else "all_patients")
        neptune_run = initialize_neptune_run(
            neptune_run_name, "mimic", tags=neptune_tags
        )
        print(f"Initialized Neptune run: {neptune_run_name}")

    run_grouped_feature_selection(
        model_config=model_config,
        training_data_config=training_data_config,
        preproc_train_csv=preproc_train_csv,
        preproc_eval_csv=preproc_eval_csv,
        cache_pytorch_datasets_path=cache_pytorch_datasets_path,
        train_pt_name=f"{model_dir_name}_train_dataset",
        eval_pt_name=f"{model_dir_name}_eval_dataset",
        overwrite_pt=True,
        k_sweep=k_sweep,
        neptune_run=neptune_run,
    )


if __name__ == "__main__":
    model_dir_name = "gru_duration_aware"  # Change as needed
    multiple_hosp_patients = True  # True if patients can have multiple hospital admissions
    model_config_path = f"/workspaces/msc-thesis-recurrent-health-modeling/_models/mimic/deep_learning/{model_dir_name}"
    if multiple_hosp_patients:
        model_config_path += "/multiple_hosp_patients"
    model_config_path += f"/{model_dir_name}_config.yaml"
    
    data_config_path = (impresources.files(configs) / "data_config.yaml")
    
    with open(data_config_path) as f:
        data_config = yaml.safe_load(f)

    training_data_config = data_config["training_data"]["mimic"]
    
    if multiple_hosp_patients:
        data_directory = training_data_config["data_directory_multiple_hosp_patients"]
    else:
        data_directory = training_data_config["data_directory"]
    
    LOG_IN_NEPTUNE = True  # Set to True to log in Neptune
    PREPROC_TRAIN_CSV = f"{data_directory}/train_tuning_preprocessed.csv"
    PREPROC_EVAL_CSV = f"{data_directory}/validation_tuning_preprocessed.csv"
    CACHE_PYTORCH_DATASETS_PATH = f"{data_directory}/pytorch_datasets"
    K_SWEEP = [
        (3, 2), (5, 3), (8, 4), (10, 5), (15, 7)
    ]
    
    neptune_run_name = f"{model_dir_name}_feat_selection"

    file_exists = check_if_file_exists(model_config_path)
    if not file_exists:
        raise FileNotFoundError(f"Model config file not found: {model_config_path}")

    if not check_if_file_exists(data_config_path):
        raise FileNotFoundError(f"Data config file not found: {data_config_path}")

    if not check_if_file_exists(PREPROC_TRAIN_CSV):
        raise FileNotFoundError(f"Preprocessed train CSV file not found: {PREPROC_TRAIN_CSV}")

    if not check_if_file_exists(PREPROC_EVAL_CSV):
        raise FileNotFoundError(f"Preprocessed eval CSV file not found: {PREPROC_EVAL_CSV}")

    main(
        neptune_run_name=neptune_run_name,
        model_config_path=model_config_path,
        k_sweep=K_SWEEP,
        training_data_config=training_data_config,
        preproc_train_csv=PREPROC_TRAIN_CSV,
        preproc_eval_csv=PREPROC_EVAL_CSV,
        cache_pytorch_datasets_path=CACHE_PYTORCH_DATASETS_PATH,
        log_in_neptune=LOG_IN_NEPTUNE,
        multiple_hosp_patients=multiple_hosp_patients,
    )
