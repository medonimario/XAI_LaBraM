import numpy as np
import os
import pickle
import json
import argparse
from tqdm import tqdm

# --- Reuse Exact Metric Logic ---
def compute_metrics(target_grads, cav_vec):
    """
    Computes Sensitivity, Cosine Similarity, and TCAV Score.
    Identical to the function used for Real Concepts.
    """
    cav_vec = cav_vec.squeeze()
    if target_grads.ndim == 1:
        target_grads = target_grads.reshape(1, -1)
        
    # 1. Sensitivities
    sensitivities = target_grads @ cav_vec
    
    # 2. Cosines
    grad_norms = np.linalg.norm(target_grads, axis=1, keepdims=True)
    grad_norms[grad_norms == 0] = 1e-9
    norm_grads = target_grads / grad_norms
    
    cav_norm = np.linalg.norm(cav_vec)
    if cav_norm == 0: cav_norm = 1e-9
    norm_cav = cav_vec / cav_norm
    
    cosines = norm_grads @ norm_cav
    
    # 3. TCAV Score
    valid_mask = np.abs(sensitivities) > 1e-9
    if np.sum(valid_mask) == 0:
        tcav_score = 0.5
    else:
        positive_count = np.sum(sensitivities[valid_mask] > 0)
        tcav_score = positive_count / np.sum(valid_mask)
        
    # Return Scalar Summaries (We don't save the full 100-sample arrays for nulls to save space)
    return float(np.mean(sensitivities)), float(np.mean(cosines)), float(tcav_score)

def main(args):
    target_layers = [int(l) for l in args.target_layers.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)
    
    grad_dir = os.path.join(args.target_gradient_dir, "target_gradients")
    
    print(f"--- Computing Metrics for 100 Null Distributions ---")
    
    # We will save one massive dictionary (or one per layer if too big)
    # Structure:
    # { layer_id: { anchor_id: { 'filter': {'scores': [], 'means': ...}, 'pattern': ... } } }
    
    for layer_id in target_layers:
        print(f"\nProcessing Layer {layer_id}...")
        
        # 1. Load Target Gradients
        grad_path = os.path.join(grad_dir, f"target_gradients_layer_{layer_id}.pkl")
        if not os.path.exists(grad_path):
            print(f"  Missing gradients, skipping.")
            continue
        with open(grad_path, 'rb') as f:
            target_grads = pickle.load(f)

        # 2. Load Null Vectors
        # This file contains the 100 Anchors x 99 Vectors
        null_vec_path = os.path.join(args.null_dir, f"null_distribution_layer_{layer_id}.pkl")
        if not os.path.exists(null_vec_path):
            print(f"  Missing null vectors, skipping.")
            continue
        with open(null_vec_path, 'rb') as f:
            null_data = pickle.load(f)

        # 3. Compute Metrics
        layer_results = {}
        
        # Iterate over the 100 Anchors
        # Using tqdm on anchors
        for anchor_idx, data in tqdm(null_data.items(), desc=f"  L{layer_id} Anchors"):
            
            anchor_res = {
                'filter': {'tcav_scores': [], 'mean_sens': [], 'mean_cos': []},
                'pattern': {'tcav_scores': [], 'mean_sens': [], 'mean_cos': []}
            }
            
            for method in ['filter', 'pattern']:
                vectors = data[method]['vectors'] # List of ~99 vectors
                
                for vec in vectors:
                    m_sens, m_cos, score = compute_metrics(target_grads, vec)
                    
                    anchor_res[method]['tcav_scores'].append(score)
                    anchor_res[method]['mean_sens'].append(m_sens)
                    anchor_res[method]['mean_cos'].append(m_cos)
            
            layer_results[anchor_idx] = anchor_res

        # 4. Save Layer Results Immediately
        # We save as PKL because JSON gets bloated with this depth
        out_file = os.path.join(args.output_dir, f"null_metrics_layer_{layer_id}.pkl")
        with open(out_file, 'wb') as f:
            pickle.dump(layer_results, f)
            
        print(f"  Saved null metrics to {out_file}")

    print("\nNull Metric Computation Complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute Metrics for Null Distributions")
    
    parser.add_argument("--target_gradient_dir", type=str, required=True)
    parser.add_argument("--null_dir", type=str, required=True, 
                        help="Directory containing null_distribution_layer_X.pkl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    
    args = parser.parse_args()
    main(args)