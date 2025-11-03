import torch
import numpy as np
import os
import pickle
import json
import argparse
import random
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tqdm import tqdm
from einops import rearrange # Tensor manipulation

# Import our custom ActivationExtractor class
from src.xai_labram.activation_extractor import ActivationExtractor
# Import model definition and utilities from the finetuning source code
from src.labram_ft import modeling_finetune, utils

# --- Plotting Function (No changes) ---
def plot_activations_and_cav(concept_activations, random_activations, classifier, pca, 
                           layer_id, cav_vector_orig, output_dir, concept_name, random_name):
    # (This function is identical to the previous script)
    plt.figure(figsize=(8, 6))
    all_activations = np.concatenate((concept_activations, random_activations), axis=0)
    concept_pca = pca.transform(concept_activations)
    random_pca = pca.transform(random_activations)

    plt.scatter(random_pca[:, 0], random_pca[:, 1], label=f'Random ({random_name})', alpha=0.6, c='blue', s=10)
    plt.scatter(concept_pca[:, 0], concept_pca[:, 1], label=f'Concept ({concept_name})', alpha=0.6, c='red', s=10)

    w = classifier.coef_[0]
    b = classifier.intercept_[0]
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    xy_pca = np.vstack([xx.ravel(), yy.ravel()]).T

    try:
        xy_orig = pca.inverse_transform(xy_pca)
        Z = classifier.decision_function(xy_orig).reshape(xx.shape)
        ax.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.7, linestyles=['-'])
    except Exception as e:
        print(f"  Skipping contour plot due to PCA inverse transform issue: {e}")

    pca_components = pca.components_
    cav_pca_projected = pca_components @ cav_vector_orig
    norm_projected = np.linalg.norm(cav_pca_projected)
    cav_pca_normalized = cav_pca_projected / norm_projected if norm_projected > 1e-6 else cav_pca_projected
    mean_pca = pca.transform(all_activations.mean(axis=0, keepdims=True))[0]
    arrow_scale = max(abs(xlim[1]-xlim[0]), abs(ylim[1]-ylim[0])) * 0.15

    ax.arrow(mean_pca[0], mean_pca[1],
             cav_pca_normalized[0] * arrow_scale, cav_pca_normalized[1] * arrow_scale,
             head_width=arrow_scale * 0.2, head_length=arrow_scale * 0.25,
             fc='green', ec='green', label='CAV direction', length_includes_head=True, zorder=5)

    plt.title(f'Layer {layer_id}: PCA Plot (Run 0)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plot_filename = f"pca_layer_{layer_id}_RUN_0.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved PCA plot for Run 0 to {plot_path}")
    plt.close()


# --- CAV Training Function (No changes) ---
def train_cav(concept_activations, random_activations, layer_id, alpha, 
              output_dir, concept_name, random_name, enable_plotting=False):
    # (This function is identical to the previous script)
    print(f"--- Training CAV for Layer {layer_id} ---")
    X = np.concatenate((concept_activations, random_activations), axis=0)
    y = np.concatenate((np.ones(concept_activations.shape[0]),
                        np.zeros(random_activations.shape[0])), axis=0)
    
    classifier = SGDClassifier(loss='log_loss', penalty='l2', alpha=alpha,
                               max_iter=1000, tol=1e-3, random_state=42, class_weight='balanced')
    classifier.fit(X, y)
    accuracy = classifier.score(X, y)
    print(f"Linear classifier accuracy: {accuracy:.4f}")

    if accuracy < 0.7 and accuracy > 0.51: print(f"Warning: Low accuracy ({accuracy:.4f}). CAV might be weak.")
    elif accuracy <= 0.51: print(f"Warning: Accuracy near/below chance ({accuracy:.4f}). CAV likely meaningless.")

    cav_unnormalized = classifier.coef_.squeeze().copy()

    if enable_plotting:
        print("  Generating PCA plot for this run...")
        pca = PCA(n_components=2, random_state=42)
        pca.fit(X)
        print(f"  PCA explained variance: {pca.explained_variance_ratio_}")
        plot_activations_and_cav(concept_activations, random_activations, classifier, pca, 
                                 layer_id, cav_unnormalized, 'reports/figures',# output_dir,
                                   concept_name, random_name)

    norm = np.linalg.norm(cav_unnormalized)
    if norm > 1e-6:
        cav_normalized = cav_unnormalized / norm
    else:
        print("Warning: CAV norm near zero.")
        cav_normalized = cav_unnormalized
    return cav_normalized, accuracy


# --- Gradient Calculation Function (No changes) ---
def get_averaged_gradient_v6(extractor, eeg_tensor_raw, layer_id):
    # (This function is identical to the previous script)
    model = extractor.model
    model.eval()
    model.zero_grad()
    gradient_value = None
    hook_handle_bwd = None

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradient_value
        if grad_output[0] is not None:
            gradient_value = grad_output[0].detach().clone()
        else:
            print(f"Warning: grad_output[0] is None in backward hook for layer {layer_id}.")

    if layer_id < 0 or layer_id >= len(model.blocks):
        print(f"Error: Invalid layer_id {layer_id} for model with {len(model.blocks)} blocks.")
        return None
    target_module = model.blocks[layer_id]
    hook_handle_bwd = target_module.register_full_backward_hook(backward_hook)

    if eeg_tensor_raw.ndim == 2: eeg_tensor = eeg_tensor_raw.unsqueeze(0)
    else: eeg_tensor = eeg_tensor_raw
    eeg_tensor = rearrange(eeg_tensor, 'B N (A T) -> B N A T', T=200)
    eeg_tensor = eeg_tensor.float().to(extractor.device) / 100

    try:
        logit = model(eeg_tensor, input_chans=extractor.input_chans)
    except Exception as e:
        print(f"Error during forward pass for gradient calculation: {e}")
        if hook_handle_bwd: hook_handle_bwd.remove()
        return None
    
    try:
        if not logit.requires_grad:
             print("Error: Logit does not require gradients. Cannot perform backward pass.")
             if hook_handle_bwd: hook_handle_bwd.remove()
             return None
        logit.backward()
    except Exception as e:
        print(f"Error during backward pass for gradient calculation: {e}")
    finally:
        if hook_handle_bwd:
            hook_handle_bwd.remove()

    if gradient_value is None:
        print(f"Warning: Backward hook did not capture gradient for layer {layer_id}.")
        return None

    if gradient_value.shape[1] > 1:
        gradient_pooled = gradient_value[:, 1:, :].mean(dim=1)
    else:
         gradient_pooled = gradient_value.mean(dim=1)

    gradient_np = gradient_pooled.squeeze().cpu().numpy()

    if np.isnan(gradient_np).any():
        print(f"Warning: NaN detected in calculated gradient for layer {layer_id}.")
        return None

    return gradient_np


# --- NEW: Fast TCAV Score Calculation ---
def calculate_tcav_score_fast(target_gradients, cav_vector):
    """
    Calculates TCAV score using pre-computed gradients.
    This is extremely fast as it's just a matrix multiplication.
    """
    if cav_vector is None:
        return np.nan, 0
    
    # Ensure 1D CAV vector
    cav_vector_1d = np.array(cav_vector).squeeze()
    if cav_vector_1d.ndim == 0:
        print("Error: Invalid CAV vector (scalar).")
        return np.nan, 0

    # Ensure 2D Gradients
    if target_gradients.ndim == 1:
        target_gradients = target_gradients.reshape(1, -1)

    # Check shape mismatch
    if target_gradients.shape[1] != cav_vector_1d.shape[0]:
        print(f"Error: Shape mismatch. Gradients {target_gradients.shape} vs CAV {cav_vector_1d.shape}")
        return np.nan, 0

    # Perform dot product for all samples at once
    # (N, D) @ (D,) -> (N,)
    sensitivities = target_gradients @ cav_vector_1d
    
    valid_sensitivities = sensitivities[~np.isnan(sensitivities)]
    if valid_sensitivities.size == 0:
        return np.nan, 0
        
    positive_count = np.sum(valid_sensitivities > 0)
    total_valid = valid_sensitivities.size
    
    tcav_score = positive_count / total_valid
    return tcav_score, positive_count, total_valid

# --- NEW: Gradient Pre-computation Function ---
def get_or_create_target_gradients(extractor, target_eeg_tensors_raw, target_layers, grad_dir):
    """
    Checks if gradients are saved to disk. If not, computes and saves them.
    Returns a dictionary of gradients, keyed by layer_id.
    """
    os.makedirs(grad_dir, exist_ok=True)
    target_gradients_by_layer = {}
    all_files_exist = True

    # First, check if all gradient files already exist
    for layer_id in target_layers:
        grad_path = os.path.join(grad_dir, f"target_gradients_layer_{layer_id}.pkl")
        if not os.path.exists(grad_path):
            all_files_exist = False
            break
    
    if all_files_exist:
        print(f"Found pre-calculated gradients in {grad_dir}. Loading...")
        for layer_id in target_layers:
            grad_path = os.path.join(grad_dir, f"target_gradients_layer_{layer_id}.pkl")
            with open(grad_path, 'rb') as f:
                target_gradients_by_layer[layer_id] = pickle.load(f)
        print("Gradients loaded.")
        return target_gradients_by_layer

    # If files don't exist, compute them
    print(f"No pre-calculated gradients found. Computing now (this may take a while)...")
    
    # Initialize dictionary of empty lists
    for layer_id in target_layers:
        target_gradients_by_layer[layer_id] = []

    # Loop through each target sample
    num_target_examples = len(target_eeg_tensors_raw)
    for i in tqdm(range(num_target_examples), desc="Pre-calculating Target Gradients"):
        eeg_tensor_raw = target_eeg_tensors_raw[i]
        
        # Loop through each layer for this sample
        for layer_id in target_layers:
            gradient_avg = get_averaged_gradient_v6(extractor, eeg_tensor_raw, layer_id)
            if gradient_avg is not None:
                target_gradients_by_layer[layer_id].append(gradient_avg)
            else:
                # Append a NaN array of the expected shape if grad fails
                # This is tricky as we don't know the shape. Let's just warn.
                print(f"Warning: Failed to get gradient for sample {i}, layer {layer_id}.")
                # We could append np.nan and handle it later, but for now let's skip
    
    # Convert lists to numpy arrays and save
    print("Saving pre-calculated gradients...")
    for layer_id in target_layers:
        grad_array = np.array(target_gradients_by_layer[layer_id])
        if grad_array.size == 0:
            print(f"Error: No gradients computed for layer {layer_id}. This layer will fail.")
            target_gradients_by_layer[layer_id] = None # Mark as failed
            continue

        print(f"  Layer {layer_id} gradient array shape: {grad_array.shape}")
        grad_path = os.path.join(grad_dir, f"target_gradients_layer_{layer_id}.pkl")
        with open(grad_path, 'wb') as f:
            pickle.dump(grad_array, f)
        
        # Overwrite list with the final array
        target_gradients_by_layer[layer_id] = grad_array
            
    return target_gradients_by_layer


# --- Main Execution Block (MODIFIED) ---
def main(args):
    load_dotenv()
    random.seed(args.seed)
    
    try:
        target_layers = [int(l) for l in args.target_layers.split(',')]
    except ValueError:
        print(f"Error: Invalid --target_layers format. Expected comma-separated integers (e.g., '0,6,11').")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    # NEW: Specific directory for pre-computed gradients
    target_grad_dir = os.path.join(args.output_dir, "target_gradients")

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Loading model...")
    try:
        extractor = ActivationExtractor(args.checkpoint_path, device=device_str)
        print(f"Model loaded successfully onto '{extractor.device}'.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load model from checkpoint: {e}")
        return

    # --- Load Raw Target EEG Data (needed for gradient calculation) ---
    target_manifest_path = os.path.join(args.manifest_dir, f'target_class_set.json')
    target_eeg_tensors_raw_all = []
    try:
        with open(target_manifest_path, 'r') as f: target_files = json.load(f)
        print(f"\nLoading {len(target_files)} raw target EEG samples for gradient calculation...")
        
        for f_path in tqdm(target_files, desc="Loading Target Samples"):
             try:
                 with open(f_path, 'rb') as f:
                     data = pickle.load(f)
                     target_eeg_tensors_raw_all.append(torch.from_numpy(data['X']))
             except Exception as load_err:
                 print(f"Warning: Skipping file {f_path} due to loading error: {load_err}")

        if not target_eeg_tensors_raw_all:
             raise ValueError("No target EEG samples were loaded successfully.")
             
        if args.max_target_samples is not None and args.max_target_samples < len(target_eeg_tensors_raw_all):
            print(f"Subsampling target examples from {len(target_eeg_tensors_raw_all)} down to {args.max_target_samples}...")
            target_eeg_tensors_raw = random.sample(target_eeg_tensors_raw_all, args.max_target_samples)
        else:
            target_eeg_tensors_raw = target_eeg_tensors_raw_all
        print(f"Using {len(target_eeg_tensors_raw)} samples for TCAV score calculation.")

    except Exception as e:
        print(f"FATAL ERROR during loading/processing raw target EEG data: {e}")
        return

    # --- Load Concept Activations (Once) ---
    print("\nLoading Concept activations...")
    concept_activations_by_layer = {}
    for layer_id in target_layers:
        concept_act_path = os.path.join(args.activation_dir, f"{args.concept_name}_layer_{layer_id}.pkl")
        try:
            with open(concept_act_path, 'rb') as f: 
                concept_activations_by_layer[layer_id] = pickle.load(f)
        except FileNotFoundError as e:
            print(f"Error loading concept activations for layer {layer_id}: {e}. This layer will be skipped.")
            
    if not concept_activations_by_layer:
        print("FATAL ERROR: No concept activations were loaded. Check paths and concept_name.")
        return

    # --- NEW: Pre-Calculate Target Gradients (Once) ---
    target_gradients_by_layer = get_or_create_target_gradients(
        extractor, 
        target_eeg_tensors_raw, 
        target_layers, 
        target_grad_dir
    )

    # --- Main Loop: Run TCAV N Times ---
    all_runs_statistics = []
    print(f"\n--- Starting TCAV Runs (Total: {args.num_runs}) ---")
    
    for run_i in tqdm(range(args.num_runs), desc="TCAV Runs"):
        print(f"\n===== RUN {run_i} / {args.num_runs - 1} =====")
        random_name = f"random_run_{run_i}"
        run_results = {"run_id": run_i, "layers": {}}
        
        enable_plotting_for_this_run = (run_i == 0 and args.plot_run_zero)
        
        for layer_id in target_layers:
            if layer_id not in concept_activations_by_layer or target_gradients_by_layer.get(layer_id) is None:
                print(f"Skipping Layer {layer_id} (missing concept activations or target gradients).")
                continue
                
            concept_acts = concept_activations_by_layer[layer_id]
            target_grads = target_gradients_by_layer[layer_id]
            
            random_act_path = os.path.join(args.activation_dir, f"{random_name}_layer_{layer_id}.pkl")
            try:
                with open(random_act_path, 'rb') as f: random_acts = pickle.load(f)
            except FileNotFoundError as e:
                print(f"Error loading random activations for layer {layer_id}, run {run_i}: {e}. Skipping layer.")
                run_results["layers"][layer_id] = {"cav_accuracy": np.nan, "tcav_score": np.nan}
                continue

            # --- 1. Train CAV ---
            cav_vector, cav_accuracy = train_cav(
                concept_acts, random_acts, layer_id, args.alpha, 
                args.output_dir, args.concept_name, f"run_{run_i}",
                enable_plotting=enable_plotting_for_this_run
            )
            
            # # --- 2. Calculate TCAV Score (Fast) ---
            # if cav_vector is not None and cav_accuracy > 0.51:
            #     tcav_score, pos_count, total_valid = calculate_tcav_score_fast(
            #         target_grads, 
            #         cav_vector
            #     )
            #     print(f"Layer {layer_id} TCAV Score: {tcav_score:.4f} ({pos_count}/{total_valid} positive)")
            # else:
            #     print(f"Skipping TCAV score for Layer {layer_id} (CAV accuracy too low: {cav_accuracy:.4f})")
            #     tcav_score = np.nan
            # --- 2. Calculate TCAV Score (Fast) ---
            if cav_vector is not None:
                tcav_score, pos_count, total_valid = calculate_tcav_score_fast(
                    target_grads, 
                    cav_vector
                )
                print(f"Layer {layer_id} TCAV Score: {tcav_score:.4f} ({pos_count}/{total_valid} positive)")
            else:
                print(f"Skipping TCAV score for Layer {layer_id} (due to invalid CAV vector)")
                tcav_score = np.nan

            # --- 3. Store Statistics ---
            run_results["layers"][layer_id] = {
                "cav_accuracy": cav_accuracy,
                "tcav_score": tcav_score
            }
            
        all_runs_statistics.append(run_results)

    # --- Print and Save Final Summary ---
    print("\n--- All TCAV Runs Complete ---")
    summary_path_json = os.path.join(args.output_dir, "tcav_statistics_all_runs.json")
    try:
        with open(summary_path_json, 'w') as f:
            json.dump(all_runs_statistics, f, indent=4)
        print(f"Saved all statistics to: {summary_path_json}")
    except Exception as e:
        print(f"Error saving statistics JSON file: {e}")

    print("\n--- Final Score Summary (Mean across all runs) ---")
    summary_stats = {}
    for layer_id in target_layers:
        scores = [run["layers"].get(layer_id, {}).get("tcav_score", np.nan) for run in all_runs_statistics]
        accuracies = [run["layers"].get(layer_id, {}).get("cav_accuracy", np.nan) for run in all_runs_statistics]
        
        valid_scores = [s for s in scores if not np.isnan(s)]
        valid_accs = [a for a in accuracies if not np.isnan(a)]

        mean_score = np.mean(valid_scores) if valid_scores else np.nan
        mean_acc = np.mean(valid_accs) if valid_accs else np.nan
        
        summary_stats[layer_id] = {"mean_tcav_score": mean_score, "mean_cav_accuracy": mean_acc}
        print(f"Layer {layer_id}: Mean TCAV = {mean_score:.4f} (Mean CAV Acc = {mean_acc:.4f})")

# --- Standard Python entry point ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run TCAV analysis over N runs and save statistics.")
    
    # --- Paths ---
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the .pth finetuned model checkpoint.")
    parser.add_argument("--activation_dir", type=str, required=True,
                        help="Directory containing all .pkl activation files (concept, target, and random_run_i).")
    parser.add_argument("--manifest_dir", type=str, required=True,
                        help="Directory containing the JSON manifest files (e.g., target_class_set.json).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save results (plots for run 0 and the final stats JSON).")

    # --- TCAV Parameters ---
    parser.add_argument("--num_runs", type=int, default=50,
                        help="Number of random sets to process (should match '02_generate_activations' output).")
    parser.add_argument("--target_layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11",
                        help="Comma-separated string of layer indices to test (e.g., '0,6,11').")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Regularization strength (alpha) for the SGD linear classifier.")
    parser.add_argument("--max_target_samples", type=int, default=100,
                        help="Limit the number of target samples for faster TCAV score calculation (set to 0 for all).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for target example subsampling.")

    # --- File Names ---
    parser.add_argument("--concept_name", type=str, default="concept_set",
                        help="The base name of the concept activation files (e.g., 'concept_set').")
    parser.add_argument("--target_class_name", type=str, default="target_class_set",
                        help="The base name of the target class manifest file (e.g., 'target_class_set').")

    # --- Options ---
    parser.add_argument("--plot_run_zero", action='store_true',
                        help="If set, generates PCA plots for the first run (run_0).")

    args = parser.parse_args()
    
    if args.max_target_samples == 0:
        args.max_target_samples = None
        print("Using all available target samples.")

    main(args)

# python -m src.xai_labram.03_run_tcav \
#    --checkpoint_path models/checkpoints/finetune_circling_v5/checkpoint-best.pth \
#    --activation_dir /work3/s204684/Thesis/circling_eeg/tcav/sanity_check_open/activations \
#    --manifest_dir /work3/s204684/Thesis/circling_eeg/tcav/sanity_check_open \
#    --output_dir /work3/s204684/Thesis/circling_eeg/tcav/sanity_check_open/results \
#    --num_runs 50 \
#    --target_layers "0,1,2,3,4,5,6,7,8,9,10,11" \
#    --concept_name "concept_set" \
#    --target_class_name "target_class_set" \
#    --max_target_samples 100 \
#    --plot_run_zero