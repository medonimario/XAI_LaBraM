import torch
import torch.nn as nn # For accessing nn.Module
import numpy as np
import os
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tqdm import tqdm # Progress bar
import math
import json
from einops import rearrange # Tensor manipulation
import random # For random sampling

# Import our custom ActivationExtractor class
from src.xai_labram.activation_extractor import ActivationExtractor
# Import model definition and utilities from the finetuning source code
from src.labram_ft import modeling_finetune, utils

# --- Configuration ---
CHECKPOINT_PATH = 'models/checkpoints/finetune_circling_v5/checkpoint-best.pth' # Path to the finetuned model
TARGET_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] # Indices of the Transformer blocks to analyze (0 to 11 for LaBraM-Base)
CONCEPT_NAME = "open" # Identifier for the concept (used in filenames)
RANDOM_NAME = "closed1" # Identifier for the random/negative examples
TARGET_CLASS_NAME = "open" # Identifier for the target class (used in filenames)
TARGET_CLASS_LABEL = 1 # The numerical label for the target class
ALPHA = 0.1 # Regularization strength (alpha) for the SGD linear classifier used to train CAVs
MAX_TARGET_SAMPLES = 100 # Limit the number of target samples for faster TCAV score calculation (set to None to use all)
RANDOM_SEED_SAMPLING = 42 # Seed for reproducibility if subsampling target examples
# ---------------------

# --- Plotting Function ---
def plot_activations_and_cav(concept_activations, random_activations, classifier, pca, layer_id, cav_vector_orig, output_dir):
    """
    Creates a 2D PCA plot of concept and random activations, the learned linear decision boundary,
    and the direction of the Concept Activation Vector (CAV).

    Args:
        concept_activations (np.ndarray): Activations for concept examples (shape: [n_concept_samples, D]).
        random_activations (np.ndarray): Activations for random examples (shape: [n_random_samples, D]).
        classifier (SGDClassifier): The trained linear classifier separating the activations.
        pca (PCA): The PCA object fitted on the combined activations.
        layer_id (int): The layer index these activations belong to.
        cav_vector_orig (np.ndarray): The *unnormalized* CAV vector (coefficients from the classifier).
        output_dir (str): Directory to save the plot image.
    """
    plt.figure(figsize=(8, 6))

    # Combine all activations (needed for PCA transform and finding data mean)
    all_activations = np.concatenate((concept_activations, random_activations), axis=0)

    # Apply PCA transformation to reduce dimensionality to 2D for plotting
    concept_pca = pca.transform(concept_activations)
    random_pca = pca.transform(random_activations)

    # Create scatter plot: blue for random, red for concept
    plt.scatter(random_pca[:, 0], random_pca[:, 1], label=f'Random ({RANDOM_NAME.capitalize()})', alpha=0.6, c='blue', s=10)
    plt.scatter(concept_pca[:, 0], concept_pca[:, 1], label=f'Concept ({CONCEPT_NAME.capitalize()})', alpha=0.6, c='red', s=10)

    # Get the linear classifier's parameters (weights w and intercept b) in the original high-dimensional space
    w = classifier.coef_[0]
    b = classifier.intercept_[0]

    # --- Plot Decision Boundary ---
    # To visualize the boundary in the 2D PCA plot, we create a grid of points in the PCA space,
    # inverse transform them back to the (approximate) original space, get the classifier's
    # decision function value for those points, and then plot the contour where the decision function is zero.
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50)) # Grid in PCA space
    xy_pca = np.vstack([xx.ravel(), yy.ravel()]).T # Flattened grid points

    try:
        # Map grid points back to original space
        xy_orig = pca.inverse_transform(xy_pca)
        # Get classifier scores for these points
        Z = classifier.decision_function(xy_orig).reshape(xx.shape)
        # Plot the Z=0 contour line (the decision boundary)
        ax.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.7, linestyles=['-'])
    except Exception as e:
        # Inverse transform might fail if grid points extend too far, handle gracefully
        print(f"  Skipping contour plot due to PCA inverse transform issue: {e}")

    # --- Plot CAV Direction Arrow ---
    # The CAV direction is given by the weight vector 'w' of the linear classifier.
    # We project this high-dimensional vector onto the 2D PCA space.
    pca_components = pca.components_ # The principal axes in original space (shape: [2, D])
    cav_pca_projected = pca_components @ cav_vector_orig # Project 'w' onto the PCA axes (shape: [2,])

    # Normalize the projected vector for consistent arrow direction plotting
    norm_projected = np.linalg.norm(cav_pca_projected)
    cav_pca_normalized = cav_pca_projected / norm_projected if norm_projected > 1e-6 else cav_pca_projected

    # Calculate the mean point of all data in PCA space (a place to start the arrow)
    mean_pca = pca.transform(all_activations.mean(axis=0, keepdims=True))[0]

    # Determine a suitable length for the arrow based on plot limits
    arrow_scale = max(abs(xlim[1]-xlim[0]), abs(ylim[1]-ylim[0])) * 0.15

    # Draw the green arrow representing the CAV direction
    ax.arrow(mean_pca[0], mean_pca[1], # Start point
             cav_pca_normalized[0] * arrow_scale, cav_pca_normalized[1] * arrow_scale, # End point offset
             head_width=arrow_scale * 0.2, head_length=arrow_scale * 0.25,
             fc='green', ec='green', label='CAV direction', length_includes_head=True, zorder=5) # zorder makes it visible

    # --- Final Plot Formatting ---
    plt.title(f'Layer {layer_id}: PCA of Activations ({CONCEPT_NAME.capitalize()} vs {RANDOM_NAME.capitalize()})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Equal scaling for X and Y axes is important for PCA plots

    # Save the plot to a file
    plot_filename = f"pca_layer_{layer_id}_{CONCEPT_NAME}_vs_{RANDOM_NAME}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150) # Use higher dpi for better quality
    print(f"  Saved PCA plot to {plot_path}")
    plt.close() # Close the plot to free memory


# --- CAV Training Function ---
def train_cav(concept_activations, random_activations, layer_id, alpha, output_dir):
    """
    Trains a linear classifier (SGDClassifier) to separate concept vs. random activations,
    saves the resulting *normalized* CAV vector, and generates a PCA plot.

    Args:
        concept_activations (np.ndarray): Activations for concept examples.
        random_activations (np.ndarray): Activations for random examples.
        layer_id (int): Identifier for the current layer.
        alpha (float): L2 regularization strength for the classifier.
        output_dir (str): Directory to save the CAV file and PCA plot.

    Returns:
        np.ndarray: The normalized CAV vector (shape: [D]), or None if training failed badly.
    """
    print(f"\n--- Training CAV for Layer {layer_id} ---")

    # 1. Prepare data: Concatenate activations and create binary labels (1=concept, 0=random)
    X = np.concatenate((concept_activations, random_activations), axis=0)
    y = np.concatenate((np.ones(concept_activations.shape[0]),
                        np.zeros(random_activations.shape[0])), axis=0)
    print(f"Total samples: {X.shape[0]} | Activation dim: {X.shape[1]}")

    # 2. Train Linear Classifier
    # We use SGDClassifier with log_loss (logistic regression), L2 penalty, specified alpha,
    # and balanced class weights to handle potential dataset size differences.
    classifier = SGDClassifier(loss='log_loss', penalty='l2', alpha=alpha,
                               max_iter=1000, tol=1e-3, random_state=42, class_weight='balanced')
    classifier.fit(X, y)
    accuracy = classifier.score(X, y) # Check accuracy on the training data
    print(f"Linear classifier accuracy: {accuracy:.4f}")

    # Add warnings based on accuracy, as low accuracy suggests the concept isn't linearly separable
    if accuracy < 0.7 and accuracy > 0.51: print(f"Warning: Low accuracy ({accuracy:.4f}). CAV might be weak.")
    elif accuracy <= 0.51: print(f"Warning: Accuracy near/below chance ({accuracy:.4f}). CAV likely meaningless.")

    # 3. Extract the CAV (unnormalized coefficients/weights of the classifier)
    # The coefficients represent the normal vector to the separating hyperplane.
    cav_unnormalized = classifier.coef_.squeeze().copy() # Squeeze removes extra dim, copy prevents modification issues

    # 4. Perform PCA and Plotting (Visual check)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X) # Fit PCA on the combined activation data for this layer
    print(f"  PCA explained variance: {pca.explained_variance_ratio_}") # How much variance PC1 and PC2 capture
    # Generate and save the plot
    plot_activations_and_cav(concept_activations, random_activations, classifier, pca, layer_id, cav_unnormalized, "reports/figures")

    # 5. Normalize the CAV vector (crucial for TCAV directional derivative)
    norm = np.linalg.norm(cav_unnormalized)
    if norm > 1e-6: # Avoid division by zero if weights are essentially zero
        cav_normalized = cav_unnormalized / norm
        print(f"Learned CAV shape: {cav_normalized.shape}, Norm: {norm:.4f}")
    else:
        print("Warning: CAV norm near zero. Using unnormalized vector (might indicate issues).")
        cav_normalized = cav_unnormalized # Fallback

    # 6. Save the *normalized* CAV vector to a file
    cav_filename = f"cav_{CONCEPT_NAME}_layer_{layer_id}.pkl"
    cav_path = os.path.join(output_dir, cav_filename)
    with open(cav_path, 'wb') as f:
        pickle.dump(cav_normalized, f)
    print(f"Saved *normalized* CAV to {cav_path}")

    return cav_normalized


# --- Gradient Calculation Function ---
def get_averaged_gradient_v6(extractor, eeg_tensor_raw, layer_id):
    """
    Calculates the gradient of the final logit with respect to the output of a specific
    Transformer block (model.blocks[layer_id]) using PyTorch hooks.

    The gradient tensor output by the hook has shape [B, SeqLen, D]. This function
    averages the gradient across the sequence length dimension (SeqLen, excluding the CLS token)
    to get a single gradient vector [D] suitable for dot product with the CAV.

    Args:
        extractor (ActivationExtractor): Initialized extractor containing the model.
        eeg_tensor_raw (torch.Tensor): The raw input EEG tensor (shape: [C, T]).
        layer_id (int): The index of the target Transformer block (e.g., 0 to 11).

    Returns:
        np.ndarray: The averaged gradient vector (shape: [D]), or None if an error occurs.
    """
    model = extractor.model # Get the underlying PyTorch model
    model.eval() # Ensure model is in evaluation mode (disables dropout etc.)
    model.zero_grad() # Clear any stale gradients from previous iterations

    # --- Setup Hook ---
    gradient_value = None # Variable to store the captured gradient
    hook_handle_bwd = None # Handle to remove the hook later

    # Define the backward hook function:
    # This function will be called automatically by PyTorch when gradients are computed
    # flowing *backwards* out of the target module (model.blocks[layer_id]).
    def backward_hook(module, grad_input, grad_output):
        # module: the layer the hook is attached to (model.blocks[layer_id])
        # grad_input: tuple of gradients w.r.t. the *inputs* of the module
        # grad_output: tuple of gradients w.r.t. the *outputs* of the module
        nonlocal gradient_value # Allow modification of the outer scope variable
        # We want d(Loss) / d(module_output), which is grad_output[0]
        if grad_output[0] is not None:
            # Detach and clone to store the gradient value safely outside the backward pass context
            gradient_value = grad_output[0].detach().clone()
        else:
            # This should ideally not happen if the output is used later in the graph
            print(f"Warning: grad_output[0] is None in backward hook for layer {layer_id}.")

    # Validate layer_id and get the target module
    if layer_id < 0 or layer_id >= len(model.blocks):
        print(f"Error: Invalid layer_id {layer_id} for model with {len(model.blocks)} blocks.")
        return None
    target_module = model.blocks[layer_id]

    # Register the backward hook *on the target module*.
    # register_full_backward_hook ensures it works correctly with DDP/multi-GPU if used later.
    hook_handle_bwd = target_module.register_full_backward_hook(backward_hook)

    # --- Preprocessing ---
    # Convert raw EEG (C, T) to the format expected by the model (B, N, A, T)
    # Adds batch dim, reshapes time, moves to device, scales values
    if eeg_tensor_raw.ndim == 2: eeg_tensor = eeg_tensor_raw.unsqueeze(0) # Add batch dim B=1
    else: eeg_tensor = eeg_tensor_raw # Assume already has batch dim
    eeg_tensor = rearrange(eeg_tensor, 'B N (A T) -> B N A T', T=200) # Reshape time into patches
    eeg_tensor = eeg_tensor.float().to(extractor.device) / 100 # Convert type, move device, scale

    # --- Forward Pass ---
    # Perform a standard forward pass through the *entire* model.
    # The hook mechanism doesn't require modifying the forward pass logic.
    try:
        logit = model(eeg_tensor, input_chans=extractor.input_chans)
    except Exception as e:
        print(f"Error during forward pass for gradient calculation: {e}")
        if hook_handle_bwd: hook_handle_bwd.remove() # Clean up hook if forward fails
        return None
    finally:
        pass # Hook removal happens after backward pass regardless

    # --- Backward Pass ---
    # Calculate gradients starting from the final output logit.
    try:
        # Check if the output requires gradients (it should if parameters require grad)
        if not logit.requires_grad:
             print("Error: Logit does not require gradients. Cannot perform backward pass.")
             if hook_handle_bwd: hook_handle_bwd.remove()
             return None

        # Initiate the backpropagation process. This triggers the registered hook.
        logit.backward()
    except Exception as e:
        print(f"Error during backward pass for gradient calculation: {e}")
        # Hook removal is crucial even if backward fails
        if hook_handle_bwd: hook_handle_bwd.remove()
        return None
    finally:
        # --- Crucial: Remove the hook ---
        # Hooks persist until explicitly removed. Failure to remove leads to memory leaks
        # and incorrect behavior in subsequent iterations.
        if hook_handle_bwd:
            hook_handle_bwd.remove()

    # --- Process Captured Gradient ---
    # Check if the hook successfully captured the gradient
    if gradient_value is None:
        print(f"Warning: Backward hook did not capture gradient for layer {layer_id}. Check model graph.")
        return None

    # gradient_value has shape [B, SeqLen, D], where SeqLen = 1 (CLS) + num_patches
    # To get a single vector comparable to our pooled CAV, we average the gradient
    # across the sequence dimension, *excluding* the CLS token's gradient (index 0).
    if gradient_value.shape[1] > 1: # Make sure there's more than just the CLS token
        # Select gradients for patch tokens (index 1 onwards) and average along sequence dim (dim=1)
        gradient_pooled = gradient_value[:, 1:, :].mean(dim=1) # Result shape: [B, D]
    else:
         # Fallback if somehow only CLS token exists (shouldn't happen with patches)
         gradient_pooled = gradient_value.mean(dim=1) # Average everything

    # Convert the resulting gradient tensor to a numpy array
    # Squeeze removes the batch dimension (assuming B=1) -> shape [D]
    gradient_np = gradient_pooled.squeeze().cpu().numpy()

    # Final check for numerical issues
    if np.isnan(gradient_np).any():
        print(f"Warning: NaN detected in calculated gradient for layer {layer_id}.")
        return None

    return gradient_np


# --- TCAV Score Calculation Function ---
def calculate_tcav_score(extractor, target_eeg_tensors_raw, cav_vector, layer_id):
    """
    Calculates the TCAV score for a given concept (represented by cav_vector)
    and target class (represented by target_eeg_tensors_raw) at a specific layer_id.

    It computes the conceptual sensitivity (dot product of gradient and CAV) for each
    target example and returns the fraction of examples with positive sensitivity.

    Args:
        extractor (ActivationExtractor): Initialized extractor (needed for model access).
        target_eeg_tensors_raw (list): List of raw EEG tensors for target class examples.
        cav_vector (np.ndarray): The *normalized* CAV vector for this layer and concept.
        layer_id (int): The index of the layer being analyzed.

    Returns:
        float: The TCAV score (between 0.0 and 1.0), or np.nan if errors occur.
    """
    print(f"\n--- Calculating TCAV Score for Layer {layer_id} (using block output grad) ---")
    positive_sensitivity_count = 0 # Counter for examples with sensitivity > 0
    num_target_examples = len(target_eeg_tensors_raw)
    sensitivities = [] # List to store sensitivity values for statistics

    print(f"Processing {num_target_examples} target examples...")

    # Ensure CAV is a 1D numpy array for dot product
    cav_vector_1d = np.array(cav_vector).squeeze()
    if cav_vector_1d.ndim == 0: # Check if CAV is invalid (e.g., scalar)
        print(f"Error: Invalid CAV vector (scalar) provided for layer {layer_id}.")
        return np.nan

    # Iterate through each raw EEG tensor of the target class
    for i in tqdm(range(num_target_examples), desc=f"Layer {layer_id} Gradients"):
        eeg_tensor_raw = target_eeg_tensors_raw[i]

        # Calculate the averaged gradient w.r.t the output of block[layer_id]
        gradient_avg = get_averaged_gradient_v6(extractor, eeg_tensor_raw, layer_id)

        # Handle potential errors during gradient calculation
        if gradient_avg is None:
             print(f"Warning: Skipping example {i} (invalid gradient received).")
             continue # Skip this example

        # Ensure gradient is also a 1D numpy array
        gradient_avg_1d = np.array(gradient_avg).squeeze()
        if gradient_avg_1d.ndim == 0:
            print(f"Warning: Skipping example {i} (gradient is scalar).")
            continue

        # Check for shape mismatch before dot product
        if gradient_avg_1d.shape != cav_vector_1d.shape:
            print(f"Warning: Skipping example {i} due to shape mismatch: "
                  f"gradient {gradient_avg_1d.shape}, cav {cav_vector_1d.shape}")
            continue

        # Calculate Conceptual Sensitivity: dot product of gradient and CAV
        conceptual_sensitivity = np.dot(gradient_avg_1d, cav_vector_1d)
        sensitivities.append(conceptual_sensitivity)

        # Increment counter if sensitivity is positive
        if conceptual_sensitivity > 0:
            positive_sensitivity_count += 1

    # --- Calculate Final TCAV Score ---
    # Check if any examples were processed successfully
    valid_examples = len(sensitivities)
    if valid_examples == 0:
        print("Error: No valid sensitivities calculated for any target examples.")
        tcav_score = np.nan
    else:
        # Score is the fraction of valid examples with positive sensitivity
        tcav_score = positive_sensitivity_count / valid_examples

    # Print summary statistics for the calculated sensitivities (optional, for debugging/analysis)
    if sensitivities:
      sensitivities_array = np.array(sensitivities)
      print(f"  Sensitivity stats: Mean={np.mean(sensitivities_array):.4f}, Std={np.std(sensitivities_array):.4f}, "
            f"Min={np.min(sensitivities_array):.4f}, Max={np.max(sensitivities_array):.4f}")
    else:
        print("  No valid sensitivities were calculated.")

    print(f"Layer {layer_id} TCAV Score ({CONCEPT_NAME.capitalize()} -> {TARGET_CLASS_NAME.capitalize()}): "
          f"{tcav_score:.4f} ({positive_sensitivity_count}/{valid_examples} positive)")
    return tcav_score


# --- Main Execution Block ---
def main():
    load_dotenv() # Load environment variables (e.g., dataset paths)
    random.seed(RANDOM_SEED_SAMPLING) # Set random seed for subsampling if used

    # --- Define File Paths ---
    dataset_root = os.getenv("CIRCLING_DATASET_PATH")
    if not dataset_root or not os.path.isdir(dataset_root):
        print(f"FATAL ERROR: CIRCLING_DATASET_PATH environment variable not set or invalid directory.")
        return
    activation_dir = os.path.join(dataset_root, "tcav_sanity_check_activations")
    cav_plot_output_dir = os.path.join(dataset_root, "tcav_sanity_check_cavs_plots")
    manifest_dir = os.path.join(dataset_root, "tcav_sanity_check_data_test_split")
    os.makedirs(cav_plot_output_dir, exist_ok=True) # Create output dir if it doesn't exist

    # --- Initialize Model via ActivationExtractor ---
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Loading model...")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"FATAL ERROR: Checkpoint file not found at '{CHECKPOINT_PATH}'")
        return
    try:
        # ActivationExtractor handles model loading based on checkpoint args
        extractor = ActivationExtractor(CHECKPOINT_PATH, device=device_str)
        print(f"Model loaded successfully onto '{extractor.device}'.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load model from checkpoint: {e}")
        return

    # --- Train Concept Activation Vectors (CAVs) ---
    trained_cavs = {} # Dictionary to store the trained CAV for each layer
    print("\n--- Starting CAV Training ---")
    for layer_id in TARGET_LAYERS:
        # Define paths for the pre-calculated activation files for this layer
        concept_act_path = os.path.join(activation_dir, f"concept_{CONCEPT_NAME}_layer_{layer_id}.pkl")
        random_act_path = os.path.join(activation_dir, f"random_{RANDOM_NAME}_layer_{layer_id}.pkl")

        # Load the activations
        try:
            with open(concept_act_path, 'rb') as f: concept_acts = pickle.load(f)
            with open(random_act_path, 'rb') as f: random_acts = pickle.load(f)
        except FileNotFoundError as e:
            print(f"Error loading activation file for layer {layer_id}: {e}. Skipping CAV training for this layer.")
            continue # Skip to the next layer if activations are missing

        # Train the CAV using the loaded activations
        cav = train_cav(concept_acts, random_acts, layer_id, ALPHA, cav_plot_output_dir)
        trained_cavs[layer_id] = cav # Store the normalized CAV
    print("\n--- CAV Training & Plotting Complete ---")

    # Check if any CAVs were successfully trained
    if not trained_cavs:
        print("FATAL ERROR: No CAVs were trained successfully. Cannot proceed.")
        return

    # --- Load Raw Target EEG Data (needed for gradient calculation) ---
    target_manifest_path = os.path.join(manifest_dir, f'target_{TARGET_CLASS_NAME}.json')
    target_eeg_tensors_raw_all = [] # List to hold all loaded raw tensors
    try:
        # Load the list of file paths for the target class examples
        with open(target_manifest_path, 'r') as f: target_files = json.load(f)

        print(f"\nLoading {len(target_files)} raw target EEG samples for gradient calculation...")
        # Load each EEG file specified in the manifest
        for f_path in tqdm(target_files, desc="Loading Target Samples"):
             try:
                 with open(f_path, 'rb') as f:
                     data = pickle.load(f)
                     # Store the raw data as a PyTorch tensor
                     target_eeg_tensors_raw_all.append(torch.from_numpy(data['X']))
             except Exception as load_err:
                 # Warn if a specific file fails to load but continue with others
                 print(f"Warning: Skipping file {f_path} due to loading error: {load_err}")

        # Check if any samples were loaded successfully
        if not target_eeg_tensors_raw_all:
             raise ValueError("No target EEG samples were loaded successfully.")

        # --- Subsample Target Examples (if configured) ---
        if MAX_TARGET_SAMPLES is not None and MAX_TARGET_SAMPLES < len(target_eeg_tensors_raw_all):
            print(f"Subsampling target examples from {len(target_eeg_tensors_raw_all)} down to {MAX_TARGET_SAMPLES}...")
            # Randomly select a subset of the loaded tensors
            target_eeg_tensors_raw = random.sample(target_eeg_tensors_raw_all, MAX_TARGET_SAMPLES)
        else:
            # Use all loaded samples if no limit is set or limit is >= total samples
            target_eeg_tensors_raw = target_eeg_tensors_raw_all
        print(f"Using {len(target_eeg_tensors_raw)} samples for TCAV score calculation.")

    except FileNotFoundError:
        print(f"FATAL ERROR: Target manifest file not found at '{target_manifest_path}'. Cannot calculate TCAV scores.")
        return
    except Exception as e:
        # Catch other potential errors during loading/sampling
        print(f"FATAL ERROR during loading/processing raw target EEG data: {e}")
        return

    # --- Calculate TCAV Scores ---
    tcav_scores = {} # Dictionary to store the final score for each layer
    results_summary = [] # List to store formatted results strings for summary
    print("\n--- Starting TCAV Score Calculation ---")
    for layer_id in TARGET_LAYERS:
        # Retrieve the pre-trained CAV for this layer
        cav_vector = trained_cavs.get(layer_id)
        if cav_vector is None:
             # This might happen if CAV training failed for this layer
             print(f"Error: CAV not found for layer {layer_id}. Skipping TCAV calculation.")
             results_summary.append(f"Layer {layer_id}: CAV not found.")
             continue # Skip to the next layer

        # Calculate the TCAV score using the loaded raw tensors, the extractor (model), and the CAV
        score = calculate_tcav_score(extractor, target_eeg_tensors_raw, cav_vector, layer_id)
        tcav_scores[layer_id] = score # Store the numerical score
        # Store a formatted string including the number of samples used
        results_summary.append(f"Layer {layer_id}: {score:.4f} (N={len(target_eeg_tensors_raw)})")

    # --- Print and Save Final Summary ---
    print("\n--- TCAV Score Calculation Complete ---")
    print("Final TCAV Scores:")
    for result in results_summary:
        print(f"  {result}")

    # Write a summary file with configuration and results
    summary_path = os.path.join(cav_plot_output_dir, "tcav_summary.txt")
    try:
        with open(summary_path, 'w') as f:
             f.write(f"TCAV Sanity Check Results ({CONCEPT_NAME.capitalize()} -> {TARGET_CLASS_NAME.capitalize()})\n")
             f.write(f"Model Checkpoint: {CHECKPOINT_PATH}\n")
             f.write(f"Analyzed Layers: {TARGET_LAYERS}\n")
             f.write(f"Concept: '{CONCEPT_NAME.capitalize()}', Random Set: '{RANDOM_NAME.capitalize()}'\n")
             f.write(f"Target Class Samples Used: {'All' if MAX_TARGET_SAMPLES is None else MAX_TARGET_SAMPLES}/{len(target_eeg_tensors_raw_all)}\n")
             f.write(f"CAV Alpha (Regularization): {ALPHA}\n")
             f.write("="*40 + "\n")
             f.write("Scores:\n")
             for result in results_summary:
                 f.write(f"  {result}\n")
             f.write("\nSanity Check Expectation:\n")
             f.write("Score should be high (~1.0) for layers where the CAV was well-defined (high linear classifier accuracy, e.g., Layer 11).\n")
             f.write("Scores should be lower or near random (~0.5) for layers where the CAV was poorly defined (low accuracy, e.g., Layers 3 & 7).\n")
        print(f"\nSummary of results saved to: {summary_path}")
    except Exception as e:
        print(f"\nError saving summary file: {e}")

# --- Standard Python entry point ---
if __name__ == '__main__':
    main() # Run the main function when the script is executed


# python -m src.xai_labram.run_tcav