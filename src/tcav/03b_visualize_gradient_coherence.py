import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import argparse
from tqdm import tqdm
import json

def compute_coherence_metrics(gradients):
    """
    Computes the geometric coherence of a set of gradients.
    
    1. Normalizes all gradients to unit vectors.
    2. Computes the Global Mean Direction.
    3. Calculates Cosine Similarity and Angle of every sample vs. Global Mean.
    """
    # 1. Normalize individual gradients
    # Shape: [N, Dim]
    norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    
    # Avoid division by zero
    norms[norms == 0] = 1e-9
    norm_grads = gradients / norms
    
    # 2. Compute Mean Direction
    # We take the mean of the NORMALIZED gradients to treat every sample equally
    # (Magnitude of the gradient shouldn't bias the direction)
    raw_mean = np.mean(norm_grads, axis=0)
    
    mean_norm = np.linalg.norm(raw_mean)
    if mean_norm < 1e-9:
        # Edge case: If gradients cancel out perfectly
        print("Warning: Mean gradient is near zero.")
        return None, None, 0
        
    global_mean_dir = raw_mean / mean_norm
    
    # 3. Compute Metrics
    # Dot product of unit vectors = Cosine Similarity
    # Shape: [N]
    cos_sims = norm_grads @ global_mean_dir
    
    # Clip numerical errors to stay in valid arccos range [-1, 1]
    cos_sims = np.clip(cos_sims, -1.0, 1.0)
    
    # Convert to degrees
    angles_deg = np.degrees(np.arccos(cos_sims))
    
    return cos_sims, angles_deg, mean_norm

def plot_layer_stats(ax_cos, ax_ang, cos_sims, angles, layer_id):
    """Plots the histograms for a single layer."""
    
    # --- Cosine Similarity Histogram ---
    mean_cos = np.mean(cos_sims)
    std_cos = np.std(cos_sims)
    
    ax_cos.hist(cos_sims, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax_cos.axvline(mean_cos, color='red', linestyle='dashed', linewidth=1)
    ax_cos.set_title(f"Layer {layer_id}: Cosine Sim (Mean={mean_cos:.2f})")
    ax_cos.set_xlim([-1.1, 1.1])
    ax_cos.set_ylabel("Count")
    ax_cos.grid(True, alpha=0.3)

    # --- Angle Histogram ---
    mean_ang = np.mean(angles)
    std_ang = np.std(angles)
    
    ax_ang.hist(angles, bins=30, color='salmon', edgecolor='black', alpha=0.7)
    ax_ang.axvline(mean_ang, color='blue', linestyle='dashed', linewidth=1)
    ax_ang.set_title(f"Layer {layer_id}: Angle (Mean={mean_ang:.1f}°)")
    ax_ang.set_xlim([0, 180])
    ax_ang.set_xlabel("Degrees")
    ax_ang.grid(True, alpha=0.3)
    
    return {
        "mean_cosine": mean_cos,
        "std_cosine": std_cos,
        "mean_angle": mean_ang,
        "std_angle": std_ang
    }

def main(args):
    target_layers = [int(l) for l in args.target_layers.split(',')]
    grad_dir = os.path.join(args.input_dir, "target_gradients")
    
    if not os.path.exists(grad_dir):
        raise FileNotFoundError(f"Gradient directory not found: {grad_dir}")

    # Prepare stats storage
    coherence_stats = {}
    
    # Setup Plotting
    # We will create one giant figure with one row per layer
    num_layers = len(target_layers)
    fig, axes = plt.subplots(num_layers, 2, figsize=(12, 4 * num_layers))
    if num_layers == 1: axes = axes.reshape(1, -1)
    
    print(f"Analyzing gradients in {grad_dir}...")
    
    for i, layer_id in enumerate(tqdm(target_layers)):
        grad_file = os.path.join(grad_dir, f"target_gradients_layer_{layer_id}.pkl")
        
        if not os.path.exists(grad_file):
            print(f"Skipping Layer {layer_id} (Not found)")
            continue
            
        # Load
        with open(grad_file, 'rb') as f:
            grads = pickle.load(f)
            
        # Compute
        cos_sims, angles, mean_magnitude_of_dir = compute_coherence_metrics(grads)
        
        if cos_sims is None:
            continue
            
        # Plot
        stats = plot_layer_stats(axes[i, 0], axes[i, 1], cos_sims, angles, layer_id)
        stats['magnitude_of_mean_vector'] = float(mean_magnitude_of_dir)
        coherence_stats[layer_id] = stats

    plt.tight_layout()
    plot_path = os.path.join(args.input_dir, "gradient_coherence_analysis.png")
    plt.savefig(plot_path)
    print(f"\nSaved plots to: {plot_path}")
    
    # --- Interpretation Hint ---
    print("\n--- Quick Interpretation ---")
    print("1. High Mean Cosine (> 0.5) / Low Mean Angle (< 60°):")
    print("   The gradients point in roughly the same direction. The class definition is stable/coherent at this layer.")
    print("2. Mean Cosine near 0.0 / Mean Angle near 90°:")
    print("   The gradients are orthogonal/random. The model may not have a unified representation of this class here.")
    print("3. Negative Mean Cosine:")
    print("   Contradictory gradients (rare for correct classifications).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Target Gradient Coherence")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Root output directory containing the 'target_gradients' folder")
    parser.add_argument("--target_layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    
    args = parser.parse_args()
    main(args)