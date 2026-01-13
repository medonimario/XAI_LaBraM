import numpy as np
import os
import pickle
import json
import argparse
from tqdm import tqdm

def compute_metrics(target_grads, cav_vec):
    """
    Computes Sensitivity, Cosine Similarity, and TCAV Score for a single CAV 
    against a set of target gradients.
    
    Args:
        target_grads: np.ndarray [N_samples, Dim]
        cav_vec: np.ndarray [Dim] (or [1, Dim])
        
    Returns:
        sensitivities: np.ndarray [N_samples]
        cosines: np.ndarray [N_samples]
        tcav_score: float (0.0 to 1.0)
    """
    # Ensure shapes
    cav_vec = cav_vec.squeeze()
    if target_grads.ndim == 1:
        target_grads = target_grads.reshape(1, -1)
        
    # 1. Compute Sensitivities (Dot Product)
    # [N, Dim] @ [Dim] -> [N]
    sensitivities = target_grads @ cav_vec
    
    # 2. Compute Cosine Similarity (Normalized Dot Product)
    # Normalize Gradients
    grad_norms = np.linalg.norm(target_grads, axis=1, keepdims=True)
    grad_norms[grad_norms == 0] = 1e-9 # Prevent div by zero
    norm_grads = target_grads / grad_norms
    
    # Normalize CAV (Just in case it wasn't strictly unit length)
    cav_norm = np.linalg.norm(cav_vec)
    if cav_norm == 0: cav_norm = 1e-9
    norm_cav = cav_vec / cav_norm
    
    cosines = norm_grads @ norm_cav
    
    # 3. Compute TCAV Score
    # Fraction of inputs where increasing the concept increases the target logit
    # (Sensitivity > 0)
    # Note: We filter out exactly 0.0 sensitivities (meaning no gradient / dead neuron)
    valid_mask = np.abs(sensitivities) > 1e-9
    
    if np.sum(valid_mask) == 0:
        tcav_score = 0.5 # Neutral if no gradients exist
    else:
        positive_count = np.sum(sensitivities[valid_mask] > 0)
        tcav_score = positive_count / np.sum(valid_mask)
        
    return sensitivities, cosines, tcav_score

def main(args):
    target_layers = [int(l) for l in args.target_layers.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Input Directories
    grad_dir = os.path.join(args.target_gradient_dir, "target_gradients")
    
    print(f"--- Computing Final Metrics for Concept: {args.concept_name} ---")
    
    # Data Structures for Saving
    # Summary: For plotting means (lightweight JSON)
    summary_data = {} 
    # Distributions: For detailed stats (heavy Pickle)
    distribution_data = {} 

    for layer_id in tqdm(target_layers, desc="Processing Layers"):
        
        # --- 1. Load Data ---
        # A. Target Gradients
        grad_path = os.path.join(grad_dir, f"target_gradients_layer_{layer_id}.pkl")
        if not os.path.exists(grad_path):
            print(f"  Missing gradients for layer {layer_id}, skipping.")
            continue
        with open(grad_path, 'rb') as f:
            target_grads = pickle.load(f) # [N_target_samples, Dim]

        # B. CAVs
        cav_path = os.path.join(args.cav_dir, f"cavs_{args.concept_name}_layer_{layer_id}.pkl")
        if not os.path.exists(cav_path):
            print(f"  Missing CAVs for layer {layer_id}, skipping.")
            continue
        with open(cav_path, 'rb') as f:
            cavs_dict = pickle.load(f) # {'filter': [100 vecs], 'pattern': [100 vecs]}

        # --- 2. Initialize Layer Storage ---
        layer_summary = {
            'filter': {'tcav_scores': [], 'mean_sensitivities': [], 'mean_cosines': []},
            'pattern': {'tcav_scores': [], 'mean_sensitivities': [], 'mean_cosines': []}
        }
        
        layer_dist = {
            'filter': {'sensitivities': [], 'cosines': []},
            'pattern': {'sensitivities': [], 'cosines': []}
        }

        # --- 3. Compute Loop ---
        for method in ['filter', 'pattern']:
            vectors = cavs_dict[method]
            
            # Loop through the 100 Random Runs
            for run_i, cav_vec in enumerate(vectors):
                
                sens, cos, score = compute_metrics(target_grads, cav_vec)
                
                # Append Summaries (Means/Scalars)
                layer_summary[method]['tcav_scores'].append(float(score))
                layer_summary[method]['mean_sensitivities'].append(float(np.mean(sens)))
                layer_summary[method]['mean_cosines'].append(float(np.mean(cos)))
                
                # Append Distributions (Arrays)
                # We assume sens/cos are numpy arrays, we can keep them as such in pickle
                layer_dist[method]['sensitivities'].append(sens)
                layer_dist[method]['cosines'].append(cos)

        # Assign to global store
        summary_data[layer_id] = layer_summary
        distribution_data[layer_id] = layer_dist

    # --- 4. Save Outputs ---
    
    # Save JSON Summary
    json_path = os.path.join(args.output_dir, f"final_metrics_summary_{args.concept_name}.json")
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
        
    # Save Pickle Distributions
    # (Optional: Split by layer if memory is an issue, but usually fits)
    pkl_path = os.path.join(args.output_dir, f"final_metrics_distributions_{args.concept_name}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(distribution_data, f)

    print(f"\nSaved Summary Metrics to: {json_path}")
    print(f"Saved Full Distributions to: {pkl_path}")
    
    # --- Quick Report ---
    print("\n--- Quick Look (Mean TCAV Scores) ---")
    for l in target_layers:
        if l in summary_data:
            f_score = np.mean(summary_data[l]['filter']['tcav_scores'])
            p_score = np.mean(summary_data[l]['pattern']['tcav_scores'])
            print(f"Layer {l}: Filter={f_score:.3f}, Pattern={p_score:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute Final TCAV Metrics (Sens, Cos, Score)")
    
    parser.add_argument("--target_gradient_dir", type=str, required=True, 
                        help="Root dir containing the 'target_gradients' folder")
    parser.add_argument("--cav_dir", type=str, required=True, 
                        help="Directory containing cavs_CONCEPT_layer_X.pkl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--concept_name", type=str, default="concept_set")
    parser.add_argument("--target_layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    
    args = parser.parse_args()
    main(args)