import numpy as np
import os
import pickle
import json
import argparse
import random
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from dotenv import load_dotenv

# --- Reuse CAV Logic (Same as Script 04) ---
# We keep these identical to ensure fair comparison

def evaluate_vector(vector, X, y):
    scores = X @ vector
    try:
        auc = roc_auc_score(y, scores)
    except ValueError:
        auc = 0.5
    if auc < 0.5: auc = 1.0 - auc
    
    # Accuracy check (Concept > Random assumption)
    preds = (scores > 0).astype(int)
    acc = accuracy_score(y, preds)
    return auc, acc

def compute_filter_cav(X_train, y_train, X_test, y_test, alpha):
    clf = SGDClassifier(penalty='l2', alpha=alpha,
                        max_iter=1000, tol=1e-3, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    vec = clf.coef_.squeeze().copy()
    norm = np.linalg.norm(vec)
    if norm > 1e-9: vec = vec / norm
    
    # Metrics
    acc_train = clf.score(X_train, y_train)
    acc_test = clf.score(X_test, y_test)
    scores_test = clf.decision_function(X_test)
    try: auc_test = roc_auc_score(y_test, scores_test)
    except: auc_test = 0.5
        
    return vec, {'acc': acc_train}, {'acc': acc_test, 'auc': auc_test}

def compute_pattern_cav(pos_acts, neg_acts, X_test=None, y_test=None):
    direction = np.mean(pos_acts, axis=0) - np.mean(neg_acts, axis=0)
    norm = np.linalg.norm(direction)
    vec = direction if norm < 1e-9 else direction / norm
        
    metrics_test = {}
    if X_test is not None and y_test is not None:
        auc, acc = evaluate_vector(vec, X_test, y_test)
        metrics_test = {'acc': acc, 'auc': auc}
    return vec, metrics_test

def main(args):
    load_dotenv()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    target_layers = [int(l) for l in args.target_layers.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"--- Computing Universal Null Distributions ---")
    print(f"Comparison Strategy: {args.num_runs} Anchors x {args.num_runs - 1} Negatives")
    
    # --- Optimization: Process Layer by Layer ---
    # Instead of loading ALL data for ALL layers (Ram Heavy), 
    # we finish one layer completely before moving to the next.
    
    for layer_id in target_layers:
        print(f"\nProcessing Layer {layer_id}...")
        
        # 1. Pre-load all random activations for this layer
        # This is much faster than reloading files inside the loop
        random_acts_cache = {}
        valid_indices = []
        
        print(f"  Loading random sets into memory...")
        for i in range(args.num_runs):
            path = os.path.join(args.random_activation_dir, f"random_run_{i}_layer_{layer_id}.pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    random_acts_cache[i] = pickle.load(f)
                valid_indices.append(i)
            else:
                pass # Silently skip missing runs
                
        if len(valid_indices) < 2:
            print(f"  Not enough data for Layer {layer_id}. Skipping.")
            continue

        # Storage for this layer
        # We will store a dictionary keyed by "anchor_idx"
        layer_null_results = {} 
        
        # 2. Double Loop: Anchor vs Negative
        # We use tqdm for progress
        pbar = tqdm(valid_indices, desc=f"  Anchors (Layer {layer_id})")
        
        for anchor_idx in pbar:
            anchor_acts = random_acts_cache[anchor_idx]
            
            anchor_res = {
                'filter': {'vectors': [], 'metrics': []},
                'pattern': {'vectors': [], 'metrics': []},
                'neg_indices': [] # Track which random run was the negative
            }
            
            for neg_idx in valid_indices:
                if anchor_idx == neg_idx: 
                    continue # Skip self-comparison
                
                neg_acts = random_acts_cache[neg_idx]
                
                # --- Prepare Data ---
                # "Concept" = Anchor (Class 1)
                # "Random"  = Negative (Class 0)
                X = np.concatenate((anchor_acts, neg_acts), axis=0)
                y = np.concatenate((np.ones(len(anchor_acts)), np.zeros(len(neg_acts))))
                
                # Train/Test Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42, stratify=y
                )
                
                # --- A. Filter Null ---
                vec_f, m_train_f, m_test_f = compute_filter_cav(X_train, y_train, X_test, y_test, args.alpha)
                
                # --- B. Pattern Null ---
                train_mask_pos = (y_train == 1)
                vec_p, m_test_p = compute_pattern_cav(
                    X_train[train_mask_pos], X_train[~train_mask_pos], X_test, y_test
                )
                
                # Store
                anchor_res['neg_indices'].append(neg_idx)
                
                anchor_res['filter']['vectors'].append(vec_f)
                anchor_res['filter']['metrics'].append({**m_test_f, 'train_acc': m_train_f['acc']})
                
                anchor_res['pattern']['vectors'].append(vec_p)
                anchor_res['pattern']['metrics'].append(m_test_p)
            
            layer_null_results[anchor_idx] = anchor_res
            
        # 3. Save Layer Results
        # We save one big pickle per layer containing all 100 null distributions
        output_pkl = os.path.join(args.output_dir, f"null_distribution_layer_{layer_id}.pkl")
        
        # To save space, we might want to separate metrics and vectors, 
        # but keeping them linked by Anchor ID is convenient.
        print(f"  Saving Null Distribution to {output_pkl}...")
        with open(output_pkl, 'wb') as f:
            pickle.dump(layer_null_results, f)
            
        # Optional: Save a lightweight JSON summary of just the metrics (for quick plotting)
        summary_json = {}
        for aid, res in layer_null_results.items():
            summary_json[aid] = {
                'filter_auc': [m['auc'] for m in res['filter']['metrics']],
                'pattern_auc': [m['auc'] for m in res['pattern']['metrics']]
            }
            
        with open(os.path.join(args.output_dir, f"null_stats_layer_{layer_id}.json"), 'w') as f:
            json.dump(summary_json, f)

    print("\nUniversal Null Computation Complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute Universal Null Distributions (Random vs Random)")
    
    parser.add_argument("--random_activation_dir", type=str, required=True, 
                        help="Directory containing random_run_X_layer_Y.pkl files")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_runs", type=int, default=100, 
                        help="Number of random files available (indices 0 to N-1)")
    parser.add_argument("--target_layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)