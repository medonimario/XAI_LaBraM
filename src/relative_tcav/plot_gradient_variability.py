import pickle
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot Gradient Variability Histograms")
    parser.add_argument("--stats_file", type=str, required=True, 
                        help="Path to the gradient_variability_stats.pkl file")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading stats from {args.stats_file}...")
    with open(args.stats_file, 'rb') as f:
        stats = pickle.load(f)

    # Get sorted layer IDs for consistent ordering
    layer_ids = sorted(stats.keys())

    for layer_id in layer_ids:
        data = stats[layer_id]
        cos_sims = data['cosine_similarities']
        angles = data['angles_degrees']

        # Create a figure with 2 subplots (side by side)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # --- Plot 1: Cosine Similarity Histogram ---
        axes[0].hist(cos_sims, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_title(f"Layer {layer_id}: Cosine Similarity to Mean Grad")
        axes[0].set_xlabel("Cosine Similarity")
        axes[0].set_ylabel("Count")
        axes[0].set_xlim([-1.05, 1.05]) # Fixed range for easy comparison
        axes[0].axvline(np.mean(cos_sims), color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {np.mean(cos_sims):.2f}')
        axes[0].legend()

        # --- Plot 2: Angle Histogram ---
        axes[1].hist(angles, bins=30, color='salmon', edgecolor='black', alpha=0.7)
        axes[1].set_title(f"Layer {layer_id}: Angle Distribution (Degrees)")
        axes[1].set_xlabel("Angle (Degrees)")
        axes[1].set_ylabel("Count")
        axes[1].set_xlim([0, 180]) # Fixed range [0, 180] for angles
        axes[1].axvline(np.mean(angles), color='blue', linestyle='dashed', linewidth=1.5, label=f'Mean: {np.mean(angles):.1f}°')
        axes[1].legend()

        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(args.output_dir, f"grad_variability_layer_{layer_id}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"Saved plot for Layer {layer_id} to {save_path}")

if __name__ == "__main__":
    main()