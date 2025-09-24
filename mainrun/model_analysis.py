import json
import pandas as pd
import matplotlib.pyplot as plt

import glob
import os
from pathlib import Path

def parse_all_logs(log_dir="./logs"):
    """
    Parse all .log files in a directory, sorted by modification time.
    
    Returns:
        dict {filename: (train_df, val_df, meta)} in time order
    """
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    
    # Sort by modification time
    log_files_sorted = sorted(log_files, key=lambda f: Path(f).stat().st_mtime)
    
    results = {}
    for fpath in log_files_sorted:
        try:
            train_df, val_df, meta = parse_log_file(fpath)
            fname = os.path.basename(fpath)
            results[fname] = (train_df, val_df, meta)
        except Exception as e:
            print(f"⚠️ Failed to parse {fpath}: {e}")
    return results

def parse_log_file(filepath: str):
    """
    Parse a JSON-lines training log file into training/validation DataFrames
    and a metadata dictionary.
    
    Args:
        filepath (str): Path to the log file.
    
    Returns:
        train_df (pd.DataFrame)
        val_df (pd.DataFrame)
        meta (dict)  # hyperparams, device, dataset, model info
    """
    train_records = []
    val_records = []
    meta = {}

    with open(filepath, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            event = entry.get("event")
            if event == "training_step":
                train_records.append({
                    "step": entry["step"],
                    "loss": entry["loss"],
                    "timestamp": entry.get("timestamp", None)
                })
            elif event == "validation_step":
                val_records.append({
                    "step": entry["step"],
                    "loss": entry["loss"],
                    "timestamp": entry.get("timestamp", None)
                })
            elif event in {"hyperparameters_configured", "device_info", "dataset_info", "model_info"}:
                # Merge all into meta dict (keep keys unique by prefixing)
                for k, v in entry.items():
                    if k != "event":
                        meta[f"{event}.{k}"] = v

    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)

    return train_df, val_df, meta

def plot_multi_models(models_dict, out_file=None):
    """
    Plot validation curves, aligned training curves, and val–train gaps.
    
    Args:
        models_dict: dict of {"name": (train_df, val_df, meta)}
        out_file: optional path (".png" or ".pdf") to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

    # Panel 1: Validation losses
    for name, (train_df, val_df, meta) in models_dict.items():
        axes[0].plot(val_df["step"], val_df["loss"], marker="o", label=name)
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title(f"Validation Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Training losses
    for name, (train_df, val_df, meta) in models_dict.items():
        axes[1].plot(train_df["step"], train_df["loss"], label=name, alpha=0.7)
    axes[1].set_ylabel("Training Loss")
    axes[1].set_title("Training Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Model info
    # Render table inside the plot
    axes[2].axis("off")
    keys = [
        "hyperparameters_configured.n_layer",
        "hyperparameters_configured.n_head",
        "hyperparameters_configured.d_model",
        "hyperparameters_configured.block_size",
        "hyperparameters_configured.batch_size",
        "hyperparameters_configured.vocab_size",
        "hyperparameters_configured.dropout",
        "hyperparameters_configured.lr",
        "hyperparameters_configured.weight_decay",
        "hyperparameters_configured.min_lr",
        "hyperparameters_configured.warmup_ratio",
        "hyperparameters_configured.epochs",
        "model_info.parameters_count",
        "device_info.device",
    ]

    rows = []
    index = []
    for name, (train_df, val_df, meta) in models_dict.items():
        row = [meta.get(k, "") for k in keys]
        # Add final val loss (last value in val_df)
        final_val = val_df["loss"].iloc[-1] if not val_df.empty else None
        row.append(round(final_val, 4) if final_val is not None else "")
        index.append(name)
        rows.append(row)

    meta_df = pd.DataFrame(rows, index=index, 
                           columns=[k.split(".")[-1] for k in keys] + ["final_val_loss"]).round(4)
    
    # Render table inside the plot
    table = axes[2].table(cellText=meta_df.values,
                          rowLabels=meta_df.index,
                          colLabels=meta_df.columns,
                          loc="center",
                          cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    axes[2].set_title("Model Meta Info Table")

    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=200, bbox_inches="tight")
        print(f" Figure saved to {out_file}")
    else:
        plt.show()
    

def main():
    results = parse_all_logs("./logs")
    plot_multi_models(results, "model_plot.pdf")


if __name__ == "__main__":
    main()
