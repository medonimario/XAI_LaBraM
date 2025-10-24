import argparse
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Assuming all provided scripts are in the same directory
from run_class_finetuning import get_dataset, get_models
from engine_for_finetuning import evaluate
import utils

def calculate_ci(metric_value, n_samples):
    """Calculates the 95% confidence interval for a proportion."""
    if n_samples == 0:
        return 0.0
    # Ensure the metric value is a proportion (between 0 and 1)
    clamped_value = np.clip(metric_value, 0, 1)
    sigma = np.sqrt(clamped_value * (1 - clamped_value) / n_samples)
    ci = 1.96 * sigma  # 95% confidence interval
    return ci

def plot_curves(log_path, output_dir, dataset_sizes):
    """
    Parses the log.txt file and creates targeted plots for specific metrics.
    """
    epochs_data = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                try:
                    epochs_data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping malformed line in log.txt: {line.strip()}")
    except IOError as e:
        print(f"Error reading log file: {e}")
        return

    if not epochs_data:
        print("Log file is empty. No curves to plot.")
        return
        
    epochs = [e['epoch'] for e in epochs_data]

    # Define the specific plots we want to create
    plot_configurations = [
        {
            "title": "Loss Curves",
            "ylabel": "Loss",
            "filename": "loss_curves.png",
            "metrics": {
                "train_loss": "Train Loss",
                "val_loss": "Validation Loss",
                "test_loss": "Test Loss"
            },
            "show_ci": False
        },
        {
            "title": "Accuracy Curves",
            "ylabel": "Balanced Accuracy",
            "filename": "accuracy_curves.png",
            "metrics": {
                "train_class_acc": "Train Accuracy",
                "val_balanced_accuracy": "Validation Balanced Accuracy",
                "test_balanced_accuracy": "Test Balanced Accuracy"
            },
            "show_ci": True
        },
        {
            "title": "ROC AUC Curves",
            "ylabel": "ROC AUC",
            "filename": "roc_auc_curves.png",
            "metrics": {
                "val_roc_auc": "Validation ROC AUC",
                "test_roc_auc": "Test ROC AUC"
            },
            "show_ci": True
        },
        {
            "title": "PR AUC Curves",
            "ylabel": "PR AUC",
            "filename": "pr_auc_curves.png",
            "metrics": {
                "val_pr_auc": "Validation PR AUC",
                "test_pr_auc": "Test PR AUC"
            },
            "show_ci": True
        }
    ]

    for config in plot_configurations:
        plt.figure(figsize=(10, 6))
        
        for key, label in config["metrics"].items():
            # Check if the metric exists in the log data
            if key in epochs_data[0]:
                values = np.array([e[key] for e in epochs_data if key in e])
                # Skip if all values are None or inf
                if np.all(np.isinf(values)) or np.all(np.isnan(values)):
                    continue

                plt.plot(epochs, values, marker='o', linestyle='-', label=label)

                # Add confidence intervals if requested and applicable
                if config["show_ci"]:
                    split = key.split('_')[0] # 'train', 'val', or 'test'
                    if split in dataset_sizes:
                        ci = calculate_ci(values, dataset_sizes[split])
                        plt.fill_between(epochs, values - ci, values + ci, alpha=0.2)

        plt.title(config["title"])
        plt.xlabel('Epoch')
        plt.ylabel(config["ylabel"])
        plt.grid(True)
        plt.legend()
        
        plot_filename = os.path.join(output_dir, config["filename"])
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved plot: {plot_filename}")


def evaluate_best_model(checkpoint_path, model_args, device):
    """
    Loads the best model, evaluates it on the test set, and saves metrics
    with confidence intervals to a text file.
    """
    output_dir = model_args.output_dir
    
    # 1. Get the dataset
    _, test_dataset, _, ch_names, metrics_list = get_dataset(model_args)
    test_dataset_size = len(test_dataset)
    
    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=int(1.5 * model_args.batch_size),
        num_workers=model_args.num_workers,
        pin_memory=True
    )

    # 2. Re-create the model
    model = get_models(model_args)

    # 3. Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = checkpoint.get('model', checkpoint.get('module', checkpoint))
    utils.load_state_dict(model, model_state_dict)
    model.to(device)
    print(f"Model loaded from {checkpoint_path} and moved to {device}.")

    # 4. Perform evaluation
    print("Running evaluation on the test set...")
    test_stats = evaluate(
        data_loader=data_loader_test, model=model, device=device,
        ch_names=ch_names, metrics=metrics_list, is_binary=(model_args.nb_classes == 1)
    )

    # 5. Save results with confidence intervals
    results_path = os.path.join(output_dir, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Evaluation Metrics for {model_args.dataset} from {checkpoint_path}\n")
        f.write(f"Test Set Size: {test_dataset_size}\n")
        f.write("="*60 + "\n")
        for key, value in test_stats.items():
            # Add CI for proportion-based metrics
            if any(k in key for k in ["accuracy", "auc"]):
                ci = calculate_ci(value, test_dataset_size)
                f.write(f"{key}: {value:.4f} ± {ci:.4f}\n")
            else: # For metrics like loss
                f.write(f"{key}: {value:.4f}\n")
    print(f"Evaluation results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description='LaBraM evaluation and plotting script.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--output_dir', required=True, type=str,
        help='Path to the checkpoint directory containing log.txt and checkpoint-best.pth'
    )
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for evaluation (e.g., "cuda", "cpu")'
    )
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        print(f"Error: Directory not found at {args.output_dir}")
        return

    checkpoint_path = os.path.join(args.output_dir, 'checkpoint-best.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Error: 'checkpoint-best.pth' not found in {args.output_dir}. Cannot evaluate model.")
        model_args_loaded = False
    else:
        try:
            model_args = torch.load(checkpoint_path, map_location='cpu')['args']
            model_args.output_dir = args.output_dir
            model_args_loaded = True
        except Exception as e:
            print(f"Could not load args from checkpoint: {e}. Cannot evaluate or plot CIs.")
            model_args_loaded = False

    log_path = os.path.join(args.output_dir, 'log.txt')
    if os.path.exists(log_path):
        dataset_sizes = {}
        if model_args_loaded:
            try:
                train_ds, test_ds, val_ds, _, _ = get_dataset(model_args)
                dataset_sizes = {'train': len(train_ds), 'test': len(test_ds), 'val': len(val_ds)}
                print(f"Dataset sizes: Train={dataset_sizes['train']}, Val={dataset_sizes['val']}, Test={dataset_sizes['test']}")
            except Exception as e:
                print(f"Could not load datasets: {e}. Plotting without confidence intervals.")
        
        print("\nPlotting training curves...")
        plot_curves(log_path, args.output_dir, dataset_sizes)
    else:
        print(f"log.txt not found in {args.output_dir}, skipping plotting.")

    if model_args_loaded:
        print("\nEvaluating best model on the test set...")
        evaluate_best_model(checkpoint_path, model_args, args.device)


if __name__ == '__main__':
    main()