from datetime import datetime
import os
import yaml
from importlib import resources as impresources

import optuna
import torch

from recurrent_health_events_prediction import configs
from recurrent_health_events_prediction.model_selection.deep_learning.utils import save_space_to_txt, save_study_artifacts
from recurrent_health_events_prediction.training.train_deep_learning_models import (
    prepare_datasets,
    train,
    evaluate
)
from recurrent_health_events_prediction.utils.general_utils import import_yaml_config
import copy

from recurrent_health_events_prediction.utils.neptune_utils import initialize_neptune_run, upload_file_to_neptune

# ===========================================================
# 1. Global setup
# ===========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_CONFIG_PATH = (impresources.files(configs) / "data_config.yaml")
MODEL_NAME = "attention_pooling"  # Options: "gru", "cross_attention_pooling"
MULTIPLE_HOSP_ADM_EVENTS = True  # Set to True if predicting multiple events per hospital admission
CACHE_PYTORCH_DATASETS_PATH = f"/workspaces/msc-thesis-recurrent-health-modeling/_models/mimic/deep_learning/{MODEL_NAME}"
MODEL_BASE_CONFIG_PATH = f"/workspaces/msc-thesis-recurrent-health-modeling/_models/mimic/deep_learning/{MODEL_NAME}"
if MULTIPLE_HOSP_ADM_EVENTS:
    CACHE_PYTORCH_DATASETS_PATH += "/multiple_hosp_patients"
    MODEL_BASE_CONFIG_PATH += "/multiple_hosp_patients"
MODEL_BASE_CONFIG_PATH += f"/{MODEL_NAME}_config.yaml"
N_TRIALS = 70
LOG_IN_NEPTUNE = True  # Set to True to log results to Neptune.ai
EARLY_STOPPING_PATIENCE = 5

base_config = import_yaml_config(MODEL_BASE_CONFIG_PATH)
with open(DATA_CONFIG_PATH) as f:
    data_config = yaml.safe_load(f)
training_data_config = data_config["training_data"]["mimic"]
if MULTIPLE_HOSP_ADM_EVENTS:
    data_directory = training_data_config["data_directory_multiple_hosp_patients"]
else:
    data_directory = training_data_config["data_directory"]

# Preload datasets once (saves time per trial)
train_dataset, validation_dataset, _, preproc_train_csv, preproc_eval_csv = prepare_datasets(
    data_directory=data_directory,
    training_data_config=training_data_config,
    model_config=base_config,
    # Raw filenames:
    raw_train_filename="train_tuning.csv",
    raw_eval_filename="validation_tuning.csv",
    # Desired .pt filenames:
    train_pt_name="train_tuning_dataset.pt",
    eval_pt_name="validation_tuning_dataset.pt",
    cache_pytorch_datasets_path=CACHE_PYTORCH_DATASETS_PATH,
    # Options:
    save_scaler_dir_path=None,
    overwrite_preprocessed=False,
    overwrite_pt=False
)

# ===========================================================
# 2. Define model-specific hyperparameter spaces
# ===========================================================

# ---- Extend the global space with model-specific defaults (optional, for UI/logging) ----
space_hyperparams = {
    # Common hyperparameters
    "learning_rate": (1e-4, 1e-2),  # log-uniform
    "weight_decay": (0.0, 0.1),     # uniform
    "batch_size": [32, 64, 128],    # categorical
    "dropout": (0.0, 0.5),          # uniform
    "hidden_size_head": [16, 32, 64, 128],  # categorical

    # Optional: defaults you can show in UIs; the sampler below still guards constraints
    "CrossAttnPoolingNet.hidden_size_seq": [32, 64, 128, 256],
    "CrossAttnPoolingNet.num_heads_candidates": [1, 2, 4, 8],
    "CrossAttnPoolingNet.use_posenc": [True, False],

    "GRUNet.hidden_size_seq": [16, 32, 64, 128, 256],
    "GRUNet.num_layers_seq": [1, 2],

    "AttentionPoolingNet.hidden_size_seq": [8, 16, 32, 64, 128],
}


def sample_hparams(trial, model_class_name):
    """Return a dictionary of sampled hyperparameters for the given model class."""
    params = {}

    # ---- Common hyperparameters ----
    lr_lo, lr_hi = space_hyperparams["learning_rate"]
    # If you're on newer Optuna, prefer: trial.suggest_float("learning_rate", lr_lo, lr_hi, log=True)
    params["learning_rate"] = trial.suggest_loguniform("learning_rate", lr_lo, lr_hi)
    params["weight_decay"] = trial.suggest_float("weight_decay", 
                                             space_hyperparams["weight_decay"][0], 
                                             space_hyperparams["weight_decay"][1])
    params["batch_size"] = trial.suggest_categorical("batch_size", space_hyperparams["batch_size"])

    do_lo, do_hi = space_hyperparams["dropout"]
    params["dropout"] = trial.suggest_float("dropout", do_lo, do_hi)

    params["hidden_size_head"] = trial.suggest_categorical(
        "hidden_size_head",
        space_hyperparams["hidden_size_head"]
    )

    # ---- Model-specific parameters ----
    if model_class_name == "GRUNet":
        hs_values = space_hyperparams["GRUNet.hidden_size_seq"]
        params["hidden_size_seq"] = trial.suggest_categorical("hidden_size_seq", hs_values)

        nl_values = space_hyperparams["GRUNet.num_layers_seq"]
        params["num_layers_seq"] = trial.suggest_categorical("num_layers_seq", nl_values)

        # (Optional) expose chosen spaces on the global dict for downstream reporting
        space_hyperparams.setdefault("hidden_size_seq", hs_values)
        space_hyperparams.setdefault("num_layers_seq", nl_values)

    elif model_class_name == "AttentionPoolingNet":
        hs_values = space_hyperparams["AttentionPoolingNet.hidden_size_seq"]
        params["hidden_size_seq"] = trial.suggest_categorical("hidden_size_seq", hs_values)
        space_hyperparams.setdefault("hidden_size_seq", hs_values)

    elif model_class_name == "CrossAttnPoolingNet":
        # 1) Sample hidden size for sequence projection (d_model)
        hs_values = space_hyperparams["CrossAttnPoolingNet.hidden_size_seq"]
        hidden_size_seq = trial.suggest_categorical("hidden_size_seq", hs_values)
        params["hidden_size_seq"] = hidden_size_seq

        # 2) Sample num_heads but enforce divisibility by hidden_size_seq
        candidate_heads = space_hyperparams["CrossAttnPoolingNet.num_heads_candidates"]
        valid_heads = [h for h in candidate_heads if hidden_size_seq % h == 0]
        # Safety: ensure we always have at least one option (we will if hs_values multiples match candidate_heads)
        if not valid_heads:
            # Fallback to 1 head if something odd happens
            valid_heads = [1]
        params["num_heads"] = trial.suggest_categorical("num_heads", valid_heads)

        # 3) Positional encoding flag
        posenc_values = space_hyperparams["CrossAttnPoolingNet.use_posenc"]
        params["use_posenc"] = trial.suggest_categorical("use_posenc", posenc_values)

        # (Optional) mirror into flat keys for downstream reporting/UI if you use them elsewhere
        space_hyperparams.setdefault("hidden_size_seq", hs_values)
        space_hyperparams.setdefault("num_heads", candidate_heads)
        space_hyperparams.setdefault("use_posenc", posenc_values)

    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

    return params


# ===========================================================
# 3. Objective function
# ===========================================================

def objective(trial, model_class_name):
    # Deep copy base config to modify safely
    model_config = copy.deepcopy(base_config)
    sampled = sample_hparams(trial, model_class_name)

    # Inject into model_config
    model_config["learning_rate"] = sampled["learning_rate"]
    model_config["batch_size"] = sampled["batch_size"]
    model_config["model_params"]["hidden_size_head"] = sampled["hidden_size_head"]
    model_config["model_params"]["dropout"] = sampled["dropout"]
    model_config["model_params"]["hidden_size_seq"] = sampled["hidden_size_seq"]

    if model_class_name == "GRUNet":
        model_config["model_params"]["num_layers_seq"] = sampled["num_layers_seq"]
    elif model_class_name == "CrossAttnPoolingNet":
        model_config["model_params"]["num_heads"] = sampled["num_heads"]
        model_config["model_params"]["use_posenc"] = sampled["use_posenc"]
    
    # Train
    model, loss_over_epochs = train(
        train_dataset,
        model_config,
        val_dataset=validation_dataset,
        trial=trial,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )

    num_epochs = len(loss_over_epochs)
    trial.set_user_attr("num_epochs", num_epochs)

    # Evaluate on test (you can also split train into train/val if you prefer)
    results, _, _, _, _ = evaluate(
        validation_dataset, model, batch_size=model_config["batch_size"]
    )

    trial.set_user_attr("auroc", results["auroc"])
    trial.set_user_attr("f1", results["f1_score"])
    trial.set_user_attr("loss", loss_over_epochs[-1])

    print(f"Trial {trial.number} with {num_epochs} epochs - AUROC: {results['auroc']:.4f}, F1: {results['f1_score']:.4f}")

    # Optuna maximizes the return value, so choose AUROC or F1
    return results["auroc"]


# ===========================================================
# 4. Run the study
# ===========================================================

def run_study(model_class_name="GRUNet", n_trials=10):
    print(f"Starting Optuna study for {model_class_name} with {n_trials} trials...")
    study = optuna.create_study(direction="maximize", study_name=f"{model_class_name}_tuning")
    study.optimize(lambda trial: objective(trial, model_class_name), n_trials=n_trials)
    print(f"Best {model_class_name} trial:")
    print(study.best_trial.params)
    print(f"Best F1: {study.best_value:.4f}")
    return study


if __name__ == "__main__":
    model_class_name = base_config.get("model_class", "GRUNet")
    
    print("Using data directory:", data_directory)
    print("Using train_csv:", preproc_train_csv)
    print("Using eval_csv:", preproc_eval_csv)
    
    model_params_dict = base_config['model_params']
    assert model_params_dict["input_size_curr"] == len(
        base_config["current_feat_cols"]
    ), f"mismatch in input_size_curr and current_feat_cols length"
    assert model_params_dict["input_size_seq"] == len(
        base_config["longitudinal_feat_cols"]
    ), f"mismatch in input_size_seq and longitudinal_feat_cols length"

    # Initialize Neptune run
    neptune_run_name = f"{MODEL_NAME}_hparam_tuning"
    neptune_tags = [MODEL_NAME, "hparam_tuning", "optuna"]
    if MULTIPLE_HOSP_ADM_EVENTS:
        neptune_tags.append("multiple_hosp_patients")
    else:
        neptune_tags.append("all_patients")
    neptune_run = (
        initialize_neptune_run(
            DATA_CONFIG_PATH, neptune_run_name, "mimic", tags=neptune_tags
        )
        if LOG_IN_NEPTUNE
        else None
    )


    print(f"\n===== Tuning {model_class_name} =====")
    print("Base config from model name:", MODEL_NAME)

    # Set up Optuna logging directory
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    optuna_run_name = f"{MODEL_NAME}_optuna_{now_str}"  # for TensorBoard logging
    base_log_dir = training_data_config.get("optuna_log_dir", "_optuna_runs")
    save_dir = os.path.join(base_log_dir, optuna_run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Optuna study artifacts will be saved to: {save_dir}\n")
    if neptune_run:
        neptune_run["study_log"] = optuna_run_name
        neptune_run["data/train_size"] = len(train_dataset)
        neptune_run["data/val_size"] = len(validation_dataset)
        neptune_run["data/train_path"] = str(preproc_train_csv)
        neptune_run["data/val_path"] = str(preproc_eval_csv)
        
    if neptune_run:
        upload_file_to_neptune(
            neptune_run,
            local_path=MODEL_BASE_CONFIG_PATH,
            neptune_base_path="artifacts",
            neptune_filename="base_config.yaml"
        )
    
    # Run the study
    study = run_study(model_class_name, n_trials=N_TRIALS)
    
    filepath = save_space_to_txt(space_hyperparams, out_dir=save_dir)
    if neptune_run:
        upload_file_to_neptune(
            neptune_run,
            local_path=filepath,
            neptune_base_path="artifacts",
            neptune_filename="hyperparams_search_space.txt"
        )

    # Save study artifacts
    save_study_artifacts(
        study,
        out_dir=save_dir,
        base_config=base_config,
        model_class_name=model_class_name,
        neptune_run=neptune_run
    )
