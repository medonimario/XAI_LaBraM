import torch
import numpy as np
import os
import pickle
import json
import argparse
import random
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tqdm import tqdm
from einops import rearrange
from sklearn.model_selection import train_test_split

from src.xai_labram.activation_extractor import ActivationExtractor

# --- Plotting Function ---
def plot_relative_tcav_analysis(
    layer_id, 
    concept_acts, 
    contrast_acts, 
    real_cav_filter, 
    real_cav_pattern,
    target_grads, 
    output_dir, 
    concept_name
):
    """
    Plots the Concept vs Contrast separation and the CAV directions relative to gradients.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Fit PCA on Concept + Contrast
    pca_data = np.concatenate([concept_acts, contrast_acts], axis=0)
    # Subsample for PCA fit if too large
    if len(pca_data) > 2000:
        indices = np.random.choice(len(pca_data), 2000, replace=False)
        pca_data = pca_data[indices]
    
    pca = PCA(n_components=2, random_state=42)
    pca.fit(pca_data)
    
    # Project Data
    proj_a = pca.transform(concept_acts) # Concept
    proj_b = pca.transform(contrast_acts) # Contrast
    
    # Subsample for plotting
    if len(proj_a) > 500: proj_a = proj_a[np.random.choice(len(proj_a), 500, replace=False)]
    if len(proj_b) > 500: proj_b = proj_b[np.random.choice(len(proj_b), 500, replace=False)]

    # Scatter
    ax.scatter(proj_b[:, 0], proj_b[:, 1], c='blue', alpha=0.2, s=15, label='Contrast (B)')
    ax.scatter(proj_a[:, 0], proj_a[:, 1], c='red', alpha=0.3, s=15, label=f'{concept_name} (A)')

    # Arrows setup
    mean_center = np.mean(np.vstack([proj_a, proj_b]), axis=0)
    arrow_scale = max(np.ptp(proj_b[:, 0]), np.ptp(proj_b[:, 1])) * 0.4 

    def get_proj_vector(vec):
        if vec is None: return np.array([0,0])
        vec_proj = vec @ pca.components_.T
        vec_norm = vec_proj / (np.linalg.norm(vec_proj) + 1e-9)
        return vec_norm * arrow_scale

    # 1. Plot Filter CAV (Green)
    v_f = get_proj_vector(real_cav_filter)
    ax.arrow(mean_center[0], mean_center[1], v_f[0], v_f[1], 
             color='lime', width=arrow_scale*0.015, label='Filter CAV')

    # 2. Plot Pattern CAV (Orange)
    v_p = get_proj_vector(real_cav_pattern)
    ax.arrow(mean_center[0], mean_center[1], v_p[0], v_p[1], 
             color='orange', width=arrow_scale*0.012, label='Pattern CAV', linestyle='--')

    # 3. Plot Mean Target Gradient (Black)
    if target_grads is not None and len(target_grads) > 0:
        mean_grad = np.mean(target_grads, axis=0)
        v_g = get_proj_vector(mean_grad)
        ax.arrow(mean_center[0], mean_center[1], v_g[0], v_g[1], 
                 color='black', width=arrow_scale*0.01, head_width=arrow_scale*0.04, label='Mean Target Grad')

    ax.set_title(f"Layer {layer_id}: Concept vs Contrast Separation")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"relative_cav_layer_{layer_id}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

def save_gradient_variability_analysis(target_grads_by_layer, output_dir):
    """
    Computes the mean gradient direction per layer and saves the distribution
    of angles (degrees) and cosine similarities of individual gradients relative to that mean.
    """
    variability_results = {}
    
    print("\n--- Analyzing Target Gradient Variability ---")
    
    for layer_id, grads in target_grads_by_layer.items():
        if grads is None or len(grads) == 0:
            continue
            
        # 1. Normalize all gradients to unit vectors
        # Shape: [N_samples, D_dim]
        norms = np.linalg.norm(grads, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9 # Avoid division by zero
        grads_norm = grads / norms
        
        # 2. Compute the Mean Direction
        # We take the mean of the raw gradients (or normalized ones), then normalize the result.
        # Usually, taking the mean of raw gradients preserves magnitude importance, 
        # but for pure directional consistency, averaging unit vectors is also valid. 
        # Here we average the normalized vectors to focus purely on direction.
        mean_vector = np.mean(grads_norm, axis=1) # This is wrong axis, should be axis=0
        mean_vector = np.mean(grads_norm, axis=0) 
        
        mean_norm = np.linalg.norm(mean_vector)
        if mean_norm < 1e-9:
            # If mean is zero vector, skipping
            continue
        mean_unit = mean_vector / mean_norm
        
        # 3. Compute Cosine Similarity (Dot product of unit vectors)
        # (N, D) @ (D,) -> (N,)
        cos_sims = grads_norm @ mean_unit
        
        # Clip to handle floating point errors slightly outside [-1, 1]
        cos_sims = np.clip(cos_sims, -1.0, 1.0)
        
        # 4. Compute Angles in Degrees
        # arccos gives radians [0, pi]
        angles_rad = np.arccos(cos_sims)
        angles_deg = np.degrees(angles_rad)
        
        variability_results[layer_id] = {
            "cosine_similarities": cos_sims,
            "angles_degrees": angles_deg
        }
        
    # Save results
    save_path = os.path.join(output_dir, "gradient_variability_stats.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(variability_results, f)
        
    print(f"Gradient variability stats saved to: {save_path}")

# --- Pattern CAV Computation ---
def compute_pattern_cav(pos_acts, neg_acts):
    mean_pos = np.mean(pos_acts, axis=0)
    mean_neg = np.mean(neg_acts, axis=0)
    direction = mean_pos - mean_neg
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return direction
    return direction / norm

def check_pattern_quality_auc(pos_acts, neg_acts, vector):
    scores_pos = pos_acts @ vector
    scores_neg = neg_acts @ vector
    y_true = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    y_scores = np.concatenate([scores_pos, scores_neg])
    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = 0.5
    if auc < 0.5:
        auc = 1.0 - auc
    return auc

# --- Filter CAV Function ---
def compute_filter_cav(pos_acts, neg_acts, alpha):
    X = np.concatenate((pos_acts, neg_acts), axis=0)
    y = np.concatenate((np.ones(pos_acts.shape[0]), np.zeros(neg_acts.shape[0])), axis=0)
    
    # Stratified split for training
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    except ValueError:
        # Fallback if too few samples
        X_train, y_train = X, y
        X_test, y_test = X, y

    clf_cav = SGDClassifier(penalty='l2', alpha=alpha, max_iter=1000, 
                            tol=1e-3, random_state=42, class_weight='balanced')
    clf_cav.fit(X_train, y_train)
    
    # Accuracy on test set
    accuracy = clf_cav.score(X_test, y_test)
    
    # Extract vector
    vec = clf_cav.coef_.squeeze().copy()
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec = vec / norm
        
    return vec, accuracy

# --- Gradient Calculation ---
def get_averaged_gradient_v6(extractor, eeg_tensor_raw, layer_id):
    model = extractor.model
    model.eval()
    model.zero_grad()
    gradient_value = None
    hook_handle_bwd = None

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradient_value
        if grad_output[0] is not None:
            gradient_value = grad_output[0].detach().clone()

    if layer_id < 0 or layer_id >= len(model.blocks):
        return None
    target_module = model.blocks[layer_id]
    hook_handle_bwd = target_module.register_full_backward_hook(backward_hook)

    if eeg_tensor_raw.ndim == 2: eeg_tensor = eeg_tensor_raw.unsqueeze(0)
    else: eeg_tensor = eeg_tensor_raw
    eeg_tensor = rearrange(eeg_tensor, 'B N (A T) -> B N A T', T=200)
    eeg_tensor = eeg_tensor.float().to(extractor.device) / 100

    try:
        logit = model(eeg_tensor, input_chans=extractor.input_chans)
        logit.backward()
    except Exception:
        if hook_handle_bwd: hook_handle_bwd.remove()
        return None
    finally:
        if hook_handle_bwd: hook_handle_bwd.remove()

    if gradient_value is None: return None

    if gradient_value.shape[1] > 1:
        gradient_pooled = gradient_value[:, 1:, :].mean(dim=1)
    else:
         gradient_pooled = gradient_value.mean(dim=1)

    gradient_np = gradient_pooled.squeeze().cpu().numpy()
    if np.isnan(gradient_np).any(): return None
    return gradient_np

# --- TCAV Scoring (Cosine Similarity) ---
def calculate_tcav_metrics(target_gradients, cav_vector):
    """
    Calculates both the TCAV Score (fraction > 0) and the Mean Cosine Similarity.
    Normalizes gradients to ensure strict cosine similarity.
    """
    if cav_vector is None: return np.nan, 0

    cav_vector_1d = np.array(cav_vector).squeeze()
    
    # Check for valid inputs
    if target_gradients is None or len(target_gradients) == 0:
        return np.nan, 0

    # Ensure gradients are 2D [Batch, Dim]
    if target_gradients.ndim == 1: 
        target_gradients = target_gradients.reshape(1, -1)
    
    # 1. Normalize Gradients to Unit Vectors (for Cosine Similarity)
    grad_norms = np.linalg.norm(target_gradients, axis=1, keepdims=True)
    grad_norms[grad_norms == 0] = 1e-9 # Avoid div/0
    normalized_grads = target_gradients / grad_norms
    
    # 2. Calculate Cosine Similarities (Range -1 to 1)
    # Since CAV is already unit length, Dot(UnitGrad, UnitCAV) = Cosine
    similarities = normalized_grads @ cav_vector_1d 
    mean_cosine_sim = np.mean(similarities[~np.isnan(similarities)])
    
    # 3. Calculate Standard TCAV Score (Fraction of Positive Sensitivities)
    # Use raw gradients for directional derivative consistency (though sign is usually same)
    sensitivities = target_gradients @ cav_vector_1d
    valid_sensitivities = sensitivities[~np.isnan(sensitivities)]
    
    if valid_sensitivities.size == 0:
        tcav_score = np.nan
    else:
        tcav_score = np.sum(valid_sensitivities > 0) / valid_sensitivities.size
        
    return tcav_score, mean_cosine_sim

# --- Gradient Pre-computation ---
def get_or_create_target_gradients(extractor, target_eeg_tensors_raw, target_layers, grad_dir):
    os.makedirs(grad_dir, exist_ok=True)
    target_gradients_by_layer = {}
    
    all_exist = True
    for layer_id in target_layers:
        if not os.path.exists(os.path.join(grad_dir, f"target_gradients_layer_{layer_id}.pkl")):
            all_exist = False; break
    
    if all_exist:
        print("Loading pre-calculated gradients...")
        for layer_id in target_layers:
            with open(os.path.join(grad_dir, f"target_gradients_layer_{layer_id}.pkl"), 'rb') as f:
                target_gradients_by_layer[layer_id] = pickle.load(f)
        return target_gradients_by_layer

    print("Computing target gradients...")
    # Initialize empty lists
    for layer_id in target_layers: target_gradients_by_layer[layer_id] = []

    # FIX: If raw list is empty, return empty immediately
    if not target_eeg_tensors_raw:
        print("Warning: No target tensors provided for gradient calculation.")
        return {l: [] for l in target_layers}

    for i in tqdm(range(len(target_eeg_tensors_raw)), desc="Gradients"):
        eeg_tensor = target_eeg_tensors_raw[i]
        for layer_id in target_layers:
            grad = get_averaged_gradient_v6(extractor, eeg_tensor, layer_id)
            if grad is not None: target_gradients_by_layer[layer_id].append(grad)
    
    for layer_id in target_layers:
        arr = np.array(target_gradients_by_layer[layer_id])
        with open(os.path.join(grad_dir, f"target_gradients_layer_{layer_id}.pkl"), 'wb') as f:
            pickle.dump(arr, f)
        target_gradients_by_layer[layer_id] = arr
            
    return target_gradients_by_layer


def main(args):
    load_dotenv()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    target_layers = [int(l) for l in args.target_layers.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)
    target_grad_dir = os.path.join(args.target_gradient_dir, "target_gradients")

    # 1. Load Model
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Loading model...")
    extractor = ActivationExtractor(args.checkpoint_path, device=device_str)

    # 2. Load Target Data & Gradients
    target_manifest = os.path.join(args.target_gradient_dir, f'target_class_set.json')
    with open(target_manifest, 'r') as f: target_files = json.load(f)
    
    target_eeg_raw = []
    
    # --- FIX for max_target_samples logic ---
    if args.max_target_samples == 0:
        samples_to_load = target_files # Load all
        print(f"Loading ALL {len(target_files)} target samples...")
    else:
        samples_to_load = target_files[:args.max_target_samples]
        print(f"Loading {len(samples_to_load)} target samples (limit={args.max_target_samples})...")

    for f_path in samples_to_load:
        with open(f_path, 'rb') as f: target_eeg_raw.append(torch.from_numpy(pickle.load(f)['X']))
    
    target_grads_by_layer = get_or_create_target_gradients(extractor, target_eeg_raw, target_layers, target_grad_dir)

    save_gradient_variability_analysis(target_grads_by_layer, args.output_dir)

    # 3. Load Activations (Concept vs Contrast)
    print("Loading Concept & Contrast activations...")
    concept_acts_by_layer = {}
    contrast_acts_by_layer = {}
    
    for layer_id in target_layers:
        try:
            # Load Concept A
            with open(os.path.join(args.concept_activation_dir, f"{args.concept_name}_layer_{layer_id}.pkl"), 'rb') as f: 
                concept_acts_by_layer[layer_id] = pickle.load(f)
            
            # Load Contrast B (Single file now)
            with open(os.path.join(args.contrast_activation_dir, f"contrast_set_layer_{layer_id}.pkl"), 'rb') as f:
                contrast_acts_by_layer[layer_id] = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load layer {layer_id}: {e}")
            pass

    # --- Initialize Storage ---
    results = {}

    print(f"\n--- Starting Relative TCAV Analysis (Permutation Test: {args.num_runs} runs) ---")
    
    for layer_id in target_layers:
        if layer_id not in concept_acts_by_layer or layer_id not in contrast_acts_by_layer:
            continue
            
        print(f"\nProcessing Layer {layer_id}...")
        
        concept_acts = concept_acts_by_layer[layer_id]
        contrast_acts = contrast_acts_by_layer[layer_id]
        target_grads = target_grads_by_layer.get(layer_id, [])
        
        # Guard against empty gradients
        if target_grads is None or len(target_grads) == 0:
            print(f"  Skipping Layer {layer_id}: No valid gradients found.")
            continue

        # --- Part 1: Real CAV Calculation (A vs B) ---
        
        # 1A. Filter CAV
        vec_filter, acc_filter = compute_filter_cav(concept_acts, contrast_acts, args.alpha)
        tcav_filter, sim_filter = calculate_tcav_metrics(target_grads, vec_filter)
        
        # 1B. Pattern CAV
        vec_pattern = compute_pattern_cav(concept_acts, contrast_acts)
        auc_pattern = check_pattern_quality_auc(concept_acts, contrast_acts, vec_pattern)
        tcav_pattern, sim_pattern = calculate_tcav_metrics(target_grads, vec_pattern)
        
        # 1C. Alignment
        alignment_score = np.dot(vec_filter, vec_pattern)

        # Store Real Results
        layer_res = {
            "real": {
                "filter": {"tcav": tcav_filter, "cosine_sim": sim_filter, "accuracy": acc_filter},
                "pattern": {"tcav": tcav_pattern, "cosine_sim": sim_pattern, "auc": auc_pattern},
                "alignment": alignment_score
            },
            "null": {
                "filter_sims": [],
                "filter_accs": [],  # <--- Added
                "filter_tcavs": [], # <--- Added
                "pattern_sims": [],
                "pattern_aucs": [], # <--- Added
                "pattern_tcavs": [] # <--- Added
            }
        }

        # --- Part 2: Permutation Test (Null Distribution) ---
        # Pool data
        X_pool = np.concatenate([concept_acts, contrast_acts], axis=0)
        n_concept = len(concept_acts)
        
        # This loop creates the Null Distribution for ALL metrics
        for i in range(args.num_runs):
            # Shuffle labels effectively by shuffling indices
            indices = np.random.permutation(len(X_pool))
            perm_concept = X_pool[indices[:n_concept]]
            perm_contrast = X_pool[indices[n_concept:]]
            
            # 2A. Null Filter CAV
            vec_null_f, acc_null_f = compute_filter_cav(perm_concept, perm_contrast, args.alpha)
            tcav_null_f, sim_null_f = calculate_tcav_metrics(target_grads, vec_null_f)
            
            layer_res["null"]["filter_sims"].append(sim_null_f)
            layer_res["null"]["filter_accs"].append(acc_null_f)
            layer_res["null"]["filter_tcavs"].append(tcav_null_f)
            
            # 2B. Null Pattern CAV
            vec_null_p = compute_pattern_cav(perm_concept, perm_contrast)
            auc_null_p = check_pattern_quality_auc(perm_concept, perm_contrast, vec_null_p)
            tcav_null_p, sim_null_p = calculate_tcav_metrics(target_grads, vec_null_p)
            
            layer_res["null"]["pattern_sims"].append(sim_null_p)
            layer_res["null"]["pattern_aucs"].append(auc_null_p)
            layer_res["null"]["pattern_tcavs"].append(tcav_null_p)

        results[layer_id] = layer_res
        
        # Plotting (Single static plot for Relative TCAV)
        plot_relative_tcav_analysis(
            layer_id, concept_acts, contrast_acts, 
            vec_filter, vec_pattern, target_grads, 
            args.output_dir, args.concept_name
        )

    # --- Save Statistics ---
    def serialize(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.floating): return float(obj)
        return obj

    with open(os.path.join(args.output_dir, "relative_tcav_results.json"), 'w') as f:
        json.dump(results, f, indent=4, default=serialize)
    print("\nSaved results JSON.")

    # --- Print Summary ---
    print("\n--- Summary (Real Cosine Similarity vs Null Mean) ---")
    for l, res in results.items():
        real_f = res['real']['filter']['cosine_sim']
        null_f_mean = np.mean(res['null']['filter_sims'])
        z_f = (real_f - null_f_mean) / (np.std(res['null']['filter_sims']) + 1e-9)
        
        real_p = res['real']['pattern']['cosine_sim']
        null_p_mean = np.mean(res['null']['pattern_sims'])
        z_p = (real_p - null_p_mean) / (np.std(res['null']['pattern_sims']) + 1e-9)
        
        align = res['real']['alignment']

        print(f"Layer {l}:")
        print(f"  Filter  | Real: {real_f:.4f} | Null: {null_f_mean:.4f} | Z-Score: {z_f:.2f}")
        print(f"  Pattern | Real: {real_p:.4f} | Null: {null_p_mean:.4f} | Z-Score: {z_p:.2f}")
        print(f"  Alignment: {align:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    
    # Renamed arguments for Relative TCAV
    parser.add_argument("--contrast_activation_dir", type=str, required=True, help="Directory with contrast activations (B)")
    parser.add_argument("--concept_activation_dir", type=str, required=True, help="Directory with concept activations (A)")
    
    parser.add_argument("--manifest_dir", type=str, required=True)
    parser.add_argument("--target_gradient_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_runs", type=int, default=100, help="Number of permutation runs")
    parser.add_argument("--target_layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max_target_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concept_name", type=str, default="concept_set")
    
    args = parser.parse_args()
    main(args)