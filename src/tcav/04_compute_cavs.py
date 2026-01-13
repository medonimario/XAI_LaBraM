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

# --- Helper: Compute Metrics ---
def evaluate_vector(vector, X, y):
    """
    Projects data X onto the vector and computes AUC and Accuracy.
    Used to evaluate both Filter and Pattern CAVs on the same data splits.
    """
    # Project data onto the vector direction
    scores = X @ vector
    
    # 1. AUC
    try:
        auc = roc_auc_score(y, scores)
    except ValueError:
        auc = 0.5
    
    # Handle inverted direction for AUC (if concept was learned 'backwards')
    if auc < 0.5:
        auc = 1.0 - auc

    # 2. Accuracy
    # For a simple projection, we assume 0 is the threshold (after centering)
    # or we fit a simple bias scalar. Here we just check sign agreement for simplicity,
    # but a better way is to use the score sign since we expect Concept > Random.
    # To be comparable to SGD, we'll check if score > threshold (determined by mean).
    # Ideally, SGD has its own predict(), but for Pattern CAV we need this manual check.
    
    # Simple thresholding at 0 (assuming centered data or bias included in vector logic)
    # or better: we assume positive scores = class 1 (Concept).
    preds = (scores > 0).astype(int)
    acc = accuracy_score(y, preds)
    
    # If the vector is flipped (AUC < 0.5 logic), we flip predictions too for Acc
    # But since we fixed AUC above, we just return the raw calculation here.
    # Note: SVM handles its own intercept, Pattern usually assumes centered data.
    
    return auc, acc

# --- 1. Filter CAV (SVM/SGD) ---
def compute_filter_cav(X_train, y_train, X_test, y_test, alpha):
    """
    Trains a linear classifier to distinguish Concept from Random.
    Returns: Normalized Vector, Train Metrics, Test Metrics.
    """
    # Train Logistic Regression (SGD)
    clf = SGDClassifier(
        penalty='l2', 
        alpha=alpha,
        max_iter=1000, 
        tol=1e-3, 
        random_state=42, 
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)
    
    # Get Vector (Weights)
    vec = clf.coef_.squeeze().copy()
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec = vec / norm
        
    # Metrics using the classifier's built-in prediction (handles intercept correctly)
    acc_train = clf.score(X_train, y_train)
    acc_test = clf.score(X_test, y_test)
    
    # For AUC we need decision function (distance to hyperplane)
    scores_test = clf.decision_function(X_test)
    try:
        auc_test = roc_auc_score(y_test, scores_test)
    except:
        auc_test = 0.5
        
    return vec, {'acc': acc_train}, {'acc': acc_test, 'auc': auc_test}

# --- 2. Pattern CAV (Difference of Means) ---
def compute_pattern_cav(pos_acts, neg_acts, X_test=None, y_test=None):
    """
    Computes Vector = Mean(Concept) - Mean(Random).
    Returns: Normalized Vector.
    Optionally evaluates on X_test/y_test if provided.
    """
    mean_pos = np.mean(pos_acts, axis=0)
    mean_neg = np.mean(neg_acts, axis=0)
    direction = mean_pos - mean_neg
    
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        vec = direction # Should be zeros
    else:
        vec = direction / norm
        
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

    # 1. Load Concept Activations
    # We load these once as they stay constant across all random runs
    print(f"Loading Concept Activations for: {args.concept_name}")
    concept_acts_by_layer = {}
    for layer_id in target_layers:
        p = os.path.join(args.concept_activation_dir, f"{args.concept_name}_layer_{layer_id}.pkl")
        if os.path.exists(p):
            with open(p, 'rb') as f:
                concept_acts_by_layer[layer_id] = pickle.load(f)
        else:
            print(f"Warning: Concept activations missing for layer {layer_id}")

    # Initialize Storage
    # Structure: results[layer_id] = { 'filter': [{'vec':..., 'stats':...}], 'pattern': ... }
    results_store = {
        l: {
            'filter_results': [],   # Will hold dicts with vectors and stats
            'pattern_results': []
        } for l in target_layers
    }

    print(f"\n--- Starting CAV Computation ({args.num_runs} Random Runs) ---")
    
    for run_i in tqdm(range(args.num_runs), desc="Runs"):
        random_name = f"random_run_{run_i}"
        
        for layer_id in target_layers:
            if layer_id not in concept_acts_by_layer: continue
            
            concept_acts = concept_acts_by_layer[layer_id]
            
            # Load Random Acts for THIS run and THIS layer
            rand_path = os.path.join(args.random_activation_dir, f"{random_name}_layer_{layer_id}.pkl")
            if not os.path.exists(rand_path):
                continue # Skip if file missing
                
            with open(rand_path, 'rb') as f:
                random_acts = pickle.load(f)

            # --- Prepare Data ---
            # X: Concatenate Concept (1) and Random (0)
            X = np.concatenate((concept_acts, random_acts), axis=0)
            y = np.concatenate((np.ones(len(concept_acts)), np.zeros(len(random_acts))))
            
            # Split Data (Train for calc, Test for validation)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )

            # --- A. Compute Filter CAV ---
            vec_filter, stats_train_f, stats_test_f = compute_filter_cav(
                X_train, y_train, X_test, y_test, args.alpha
            )
            
            results_store[layer_id]['filter_results'].append({
                'run_id': run_i,
                'vector': vec_filter,
                'metrics': {**{f'train_{k}': v for k,v in stats_train_f.items()},
                            **{f'test_{k}': v for k,v in stats_test_f.items()}}
            })

            # --- B. Compute Pattern CAV ---
            # 1. Compute vector using ALL data (or just train, but Pattern is usually descriptive stats)
            # To be strictly comparable to Filter train/test split, we should compute Mean on X_train.
            # If we compute on All, we leak info into Test.
            # Let's compute on X_train to ensure the Test metrics are valid generalization scores.
            
            # Split X_train back into Pos/Neg for mean calculation
            train_mask_pos = (y_train == 1)
            train_pos = X_train[train_mask_pos]
            train_neg = X_train[~train_mask_pos]
            
            vec_pattern, stats_test_p = compute_pattern_cav(
                train_pos, train_neg, X_test, y_test
            )
            
            # Also get train accuracy for pattern just for logging
            _, acc_train_p = evaluate_vector(vec_pattern, X_train, y_train)

            results_store[layer_id]['pattern_results'].append({
                'run_id': run_i,
                'vector': vec_pattern,
                'metrics': {
                    'train_acc': acc_train_p,
                    'test_acc': stats_test_p['acc'],
                    'test_auc': stats_test_p['auc']
                }
            })

    # --- Save Results ---
    print("\nSaving vectors and metrics...")
    
    # We will split saving into two files per layer to keep file sizes manageable 
    # and separate binary vectors from JSON-readable metrics.
    
    metrics_summary = {}

    for layer_id in target_layers:
        if not results_store[layer_id]['filter_results']:
            continue
            
        layer_vectors = {'filter': [], 'pattern': []}
        layer_metrics = {'filter': [], 'pattern': []}
        
        # Unpack data
        for res in results_store[layer_id]['filter_results']:
            layer_vectors['filter'].append(res['vector'])
            layer_metrics['filter'].append(res['metrics'])
            
        for res in results_store[layer_id]['pattern_results']:
            layer_vectors['pattern'].append(res['vector'])
            layer_metrics['pattern'].append(res['metrics'])

        # 1. Save Vectors (Pickle)
        vec_path = os.path.join(args.output_dir, f"cavs_{args.concept_name}_layer_{layer_id}.pkl")
        with open(vec_path, 'wb') as f:
            pickle.dump(layer_vectors, f)
            
        # 2. Collect Metrics for JSON
        metrics_summary[layer_id] = layer_metrics

    # Save Metrics (JSON)
    json_path = os.path.join(args.output_dir, f"metrics_{args.concept_name}.json")
    with open(json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)

    print(f"Vectors saved to: {args.output_dir}/cavs_*.pkl")
    print(f"Metrics saved to: {json_path}")
    
    # --- Quick Report ---
    print("\n--- Quick Metric Summary (Test AUC Mean) ---")
    for l, metrics in metrics_summary.items():
        mean_auc_filter = np.mean([m['test_auc'] for m in metrics['filter']])
        mean_auc_pattern = np.mean([m['test_auc'] for m in metrics['pattern']])
        print(f"Layer {l}: Filter AUC={mean_auc_filter:.3f} | Pattern AUC={mean_auc_pattern:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute Filter and Pattern CAVs")
    
    parser.add_argument("--concept_activation_dir", type=str, required=True)
    parser.add_argument("--random_activation_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    parser.add_argument("--concept_name", type=str, default="concept_set")
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--target_layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--alpha", type=float, default=0.1, help="Regularization for SGDClassifier")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)