import os
from importlib import resources as impresources
from importlib import import_module
import optuna
import yaml
from typing import Optional, Tuple
from datetime import datetime
import neptune

import math
import torch

from recurrent_health_events_prediction import configs
from recurrent_health_events_prediction.datasets.HospReadmDataset import (
    HospReadmDataset,
)
from recurrent_health_events_prediction.training.train_deep_learning_models import evaluate, prepare_datasets
from recurrent_health_events_prediction.training.utils import plot_loss_function_epochs
from recurrent_health_events_prediction.training.utils_deep_learning import (
    compute_val_auroc,
    init_scheduler,
)
from recurrent_health_events_prediction.utils.general_utils import import_yaml_config, save_yaml_config
from recurrent_health_events_prediction.utils.neptune_utils import (
    add_model_config_to_neptune,
    add_plotly_plots_to_neptune_run,
    initialize_neptune_run,
    upload_file_to_neptune,
    upload_model_to_neptune,
)

def train_head_only(
    train_dataset: HospReadmDataset,
    model_config: dict,
    ModelClass: Optional[torch.nn.Module] = None,
    neptune_run: Optional[neptune.Run] = None,
    writer=None,
    # --- optional for HPO / validation ---
    val_dataset: Optional[HospReadmDataset] = None,
    trial=None,  # Optuna trial or None
    early_stopping_patience: Optional[int] = None,
) -> Tuple[torch.nn.Module, list[float]]:
    """
    Train the model on CPU. If `val_dataset` is provided, computes AUROC each epoch
    using `compute_val_auroc(model, val_loader)` (no call to evaluate()) and logs
    it to Neptune. Optionally prunes with Optuna and early-stops with patience.
    Returns (model, loss_over_epochs) just like before.
    """

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=model_config["batch_size"], shuffle=True
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=model_config["batch_size"], shuffle=False
        )
        print(f"Validation samples: {len(val_dataset)}")

    # Model class resolution (unchanged)
    if ModelClass is not None:
        print(f"\nUsing provided model class: {ModelClass.__name__}")
    else:
        model_class_name = model_config.get("model_class", "GRUNet")
        mod = import_module("recurrent_health_events_prediction.model.RecurrentHealthEventsDL")
        try:
            ModelClass = getattr(mod, model_class_name)
            print(f"\nUsing model class: {model_class_name}")
        except AttributeError:
            raise ImportError(
                f"Model class '{model_class_name}' not found in RecurrentHealthEventsDL"
            )

    model: torch.nn.Module = ModelClass(**model_config["model_params"])

    print("\nModel initialized and ready for training.")
    print("Model parameters:")
    for key, value in model_config["model_params"].items():
        print(f"  {key}: {value}")

    has_attention: bool = model.has_attention()
    model.train()
    
    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze only the classifier head
    for p in model.classifier_head.parameters():
        p.requires_grad = True

    # Loss / Optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=model_config["learning_rate"],
        weight_decay=model_config.get("weight_decay", 0.0),
    )
    
    scheduler = init_scheduler(optimizer, model_config, val_dataset)

    loss_over_epochs: list[float] = []

    max_num_epochs = model_config.get("num_epochs", 100)

    print("\nStarting training...")
    print(f"Max epochs: {max_num_epochs}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Batch size: {model_config['batch_size']}")
    print(f"Learning rate: {model_config['learning_rate']}")
    print(f"Weight decay: {model_config.get('weight_decay', 0.0)}")
    print(f"LR Scheduler: {model_config.get('lr_scheduler')}")
    print("Optimizer: AdamW")
    print("Loss function: BCEWithLogitsLoss\n")

    # Early stopping bookkeeping (only active if both val and patience provided)
    best_state = None
    best_val = -math.inf
    epochs_no_improve = 0
    use_es = (val_loader is not None) and (early_stopping_patience is not None)

    # Training loop
    for epoch in range(max_num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            x_current, x_past, mask_past, labels = batch
            labels = labels.float()  # BCEWithLogitsLoss expects float targets

            optimizer.zero_grad()

            if has_attention: # attention models return (logits, attention_scores)
                outputs_logits, _ = model(x_current, x_past, mask_past)
                outputs_logits = outputs_logits.squeeze(-1)
            else:
                outputs_logits = model(x_current, x_past, mask_past).squeeze(-1)

            loss = criterion(outputs_logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_over_epochs.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{model_config['num_epochs']}, Loss: {avg_epoch_loss:.6f}")
        if neptune_run:
            neptune_run["train/loss"].log(avg_epoch_loss)

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar("train/loss", avg_epoch_loss, epoch + 1)
            for name, param in model.named_parameters():
                if param.requires_grad and param.data is not None:
                    writer.add_histogram(f"weights/{name}", param.data, epoch + 1)
                if param.grad is not None:
                    writer.add_histogram(f"grads/{name}", param.grad, epoch + 1)

        # ----- Optional: validation AUC, pruning, early stopping -----
        if val_loader is not None:
            # uses your external helper
            val_auroc = compute_val_auroc(model, val_loader)
            print(f"  -> Val AUROC: {val_auroc:.6f}")
            if neptune_run:
                neptune_run["train/val_auroc"].log(val_auroc)

            # Optuna pruning hook (if trial provided)
            if trial is not None:
                trial.report(val_auroc, step=epoch)
                if trial.should_prune():
                    print("  -> Trial pruned by Optuna.")
                    raise optuna.TrialPruned()

            # Early stopping with best-weight restore
            if use_es:
                if val_auroc > best_val:
                    best_val = val_auroc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_stopping_patience:
                        if neptune_run:
                            neptune_run["train/early_stopping_epoch"] = epoch + 1
                            neptune_run["train/best_val_auroc"] = best_val
                        print(f"Early stopping at epoch {epoch + 1}. Best Val AUROC: {best_val:.6f}")
                        break
    
        # Step the scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric = val_auroc if val_loader is not None else avg_epoch_loss
                scheduler.step(metric)
            else:
                scheduler.step()
        
        # Optional simple LR logging
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  -> Current LR: {current_lr:.6g}")
        if neptune_run:
            neptune_run["train/lr"].log(current_lr)

    print("\nTraining complete.")

    # Total number of epochs run
    print(f"Total epochs run: {len(loss_over_epochs)}")
    if neptune_run:
        neptune_run["train/total_epochs"] = len(loss_over_epochs)

    # Restore best weights if early stopping was used
    if use_es and best_state is not None:
        model.load_state_dict(best_state)

    return model, loss_over_epochs


def main(
    model_config_path: str,
    overwrite_data_files: bool = False,
    cache_pytorch_datasets_path: Optional[str] = None,
    log_in_neptune: bool = False,
    neptune_tags: Optional[list[str]] = None,
    neptune_run_name: str = "deep_learning_model_run",
    tensorboard_run_name: str = "deep_learning_model_run",
    pretrained_filename: Optional[str] = None,
):
    print("Starting training script...")
    print("Logging in Neptune:", log_in_neptune)

    data_config_path = (impresources.files(configs) / "data_config.yaml")

    with open(data_config_path) as f:
        data_config = yaml.safe_load(f)

    training_data_config = data_config["training_data"]["mimic"]

    data_directory = "/workspaces/msc-thesis-recurrent-health-modeling/data/mimic-iii-preprocessed/copd_hf_renal_diabetes/train_test/first_hosp_only"

    model_config = import_yaml_config(model_config_path)
    model_config_dir_path = os.path.dirname(model_config_path)
    model_name = model_config.get("model_name", "unknown_model")

    print(f"Using model config from: {model_config_path}")
    print(f"Model name: {model_name}")
    print(f"Data directory: {data_directory}")

    model_params_dict = model_config['model_params']
    assert model_params_dict["input_size_curr"] == len(
        model_config["current_feat_cols"]
    ), f"mismatch in input_size_curr and current_feat_cols length"
    assert model_params_dict["input_size_seq"] == len(
        model_config["longitudinal_feat_cols"]
    ), f"mismatch in input_size_seq and longitudinal_feat_cols length"

    neptune_run = (
        initialize_neptune_run(
            neptune_run_name, "mimic", tags=neptune_tags
        )
        if log_in_neptune
        else None
    )

    if neptune_run:
        add_model_config_to_neptune(neptune_run, model_config)
        neptune_run["train/data_directory"] = data_directory
        neptune_run["tensorboard/run_name"] = tensorboard_run_name

    if cache_pytorch_datasets_path is None:
        cache_pytorch_datasets_path = model_config_dir_path

    # Prepare train and validation datasets (load existing or create + save new)
    train_dataset, val_dataset, _, _, _ = prepare_datasets(
        data_directory=data_directory,
        training_data_config=training_data_config,
        model_config=model_config,
        # Raw filenames:
        raw_train_filename="train_first_hosp_only_final.csv",
        raw_eval_filename="validation_first_hosp_only_final.csv",
        # Desired .pt filenames:
        train_pt_name="train_first_hosp_only_final.pt",
        eval_pt_name="validation_first_hosp_only_final.pt",
        cache_pytorch_datasets_path=cache_pytorch_datasets_path,
        # Options:
        save_scaler_dir_path=None,
        overwrite_preprocessed=overwrite_data_files,
        overwrite_pt=overwrite_data_files
    )

    # Create TensorBoard writer (log dir name matches your run name)
    base_log_dir = training_data_config.get("tensorboard_log_dir", "_runs")
    log_dir = os.path.join(base_log_dir, tensorboard_run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save model_config as YAML in log_dir
    model_config_yaml_outpath = os.path.join(log_dir, "model_config.yaml")
    save_yaml_config(model_config, model_config_yaml_outpath)
    print(f"Saved model config to {model_config_yaml_outpath}")

    # Train model
    model, loss_epochs = train_head_only(
        train_dataset, model_config, neptune_run=neptune_run,
        val_dataset=val_dataset, early_stopping_patience=5
    )

    fig = plot_loss_function_epochs(
        loss_epochs,
        num_samples=len(train_dataset),
        batch_size=model_config["batch_size"],
        learning_rate=model_config["learning_rate"],
        save_fig_dir=log_dir,
    )

    print("Saving pretrained model...")
    # Save the trained model
    model_save_path = os.path.join(
        log_dir, f"{pretrained_filename}.pth"
    )
    torch.save(model.state_dict(), model_save_path)
    print(f"Pretrained model saved to {model_save_path}")

    # Model train metrics
    train_metrics, _, _, _, _ = evaluate(
        train_dataset, model, batch_size=model_config["batch_size"]
    )
    print("Training metrics: ", train_metrics)

    if neptune_run:
        upload_model_to_neptune(neptune_run, model_save_path)
        add_plotly_plots_to_neptune_run(neptune_run, fig, "loss_per_epoch", "train")
        for metric_name, metric_value in train_metrics.items():
            if metric_name != "confusion_matrix":
                neptune_run[f"train/{metric_name}"] = metric_value

    if neptune_run:
        upload_file_to_neptune(
            neptune_run,
            model_config_yaml_outpath,
            neptune_base_path="artifacts/config",
            neptune_filename="model_config.yaml"
        )
        neptune_run["num_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        neptune_run["training_data_stats/train/num_samples"] = len(train_dataset)
        neptune_run.stop()

    print("Training and evaluation complete.")


if __name__ == "__main__":
    print("Imports complete. Running main...")
    model_dir_name = "attention_pooling_query_curr_two_stage"  # Change as needed
    multiple_hosp_patients = False  # True if patients can have multiple hospital admissions
    model_config_path = f"/workspaces/msc-thesis-recurrent-health-modeling/_models/mimic/deep_learning/{model_dir_name}"
    if multiple_hosp_patients:
        model_config_path += "/multiple_hosp_patients"
    model_config_path += f"/{model_dir_name}_config.yaml"
    overwrite_preprocessed = False

    LOG_IN_NEPTUNE = True  # Set to True to log in Neptune
    neptune_run_name = f"attention_pooling_query_curr_head_pretrained_run"
    neptune_tags = ["deep_learning", "mimic"]
    if multiple_hosp_patients:
        neptune_tags.append("multiple_hosp_patients")
    else:
        neptune_tags.append("all_patients")
    
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_run_name = f"{model_dir_name}_{now_str}"  # for TensorBoard logging

    # Check if the provided paths exist
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model config path does not exist: {model_config_path}")
    
    if multiple_hosp_patients:
        save_pytorch_datasets_path = os.path.dirname(model_config_path)
    else:
        save_pytorch_datasets_path = None  # Saves in the same dir as model_config_path

    main(
        model_config_path=model_config_path,
        cache_pytorch_datasets_path=save_pytorch_datasets_path,
        overwrite_data_files=overwrite_preprocessed,
        log_in_neptune=LOG_IN_NEPTUNE,
        neptune_run_name=neptune_run_name,
        neptune_tags=neptune_tags,
        tensorboard_run_name=tensorboard_run_name,
        pretrained_filename="attention_pooling_query_curr_head_pretrained",
    )
