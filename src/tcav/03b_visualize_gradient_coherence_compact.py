import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import argparse
from tqdm import tqdm
import pandas as pd
import seaborn as sns

# --- Styling ---
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
HIST_COLOR_COS = '#A3D5FF'  # Pastel Blue
HIST_COLOR_ANG = '#FF7886'  # Pastel Red
MEAN_LINE_COLOR = '#333333' # Dark Gray for contrast

def compute_coherence_metrics(gradients):
    """
    Computes the geometric coherence of a set of gradients.
    """
    # 1. Normalize individual gradients
    norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    norm_grads = gradients / norms
    
    # 2. Compute Mean Direction
    raw_mean = np.mean(norm_grads, axis=0)
    mean_norm = np.linalg.norm(raw_mean)
    
    if mean_norm < 1e-9:
        return None, None, 0
        
    global_mean_dir = raw_mean / mean_norm
    
    # 3. Compute Metrics
    cos_sims = norm_grads @ global_mean_dir
    cos_sims = np.clip(cos_sims, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cos_sims))
    
    return cos_sims, angles_deg, mean_norm

def load_data_into_dataframe(args):
    """
    Loads gradients and aggregates into a DataFrame.
    """
    target_layers = [int(l) for l in args.target_layers.split(',')]
    grad_dir = os.path.join(args.input_dir, "target_gradients")
    
    if not os.path.exists(grad_dir):
        raise FileNotFoundError(f"Gradient directory not found: {grad_dir}")

    print(f"Loading gradients from {grad_dir}...")
    records = []

    for layer_id in tqdm(target_layers):
        grad_file = os.path.join(grad_dir, f"target_gradients_layer_{layer_id}.pkl")
        
        if not os.path.exists(grad_file):
            continue
            
        with open(grad_file, 'rb') as f:
            grads = pickle.load(f)
            
        cos_sims, angles, _ = compute_coherence_metrics(grads)
        
        if cos_sims is None:
            continue
            
        for c, a in zip(cos_sims, angles):
            records.append({'layer': layer_id, 'cosine': c, 'angle': a})
            
    return pd.DataFrame(records)

def plot_one_sided_histograms(ax, df, metric_col, color, title, y_label, y_limits):
    """
    Draws a vertical, one-sided histogram for each layer.
    """
    layers = sorted(df['layer'].unique())
    
    # --- Config ---
    bin_count = 100          # More bins for finer detail
    max_bar_width = 0.85    # Occupy 85% of the space between integers
    
    # Offset to center the visual weight. 
    # If we start at 'layer - 0.4', the histogram grows to the right towards 'layer + 0.45'
    start_offset = -0.4 

    for layer in layers:
        subset = df[df['layer'] == layer][metric_col].values
        if len(subset) == 0: continue
        
        # 1. Calculate Histogram
        counts, bin_edges = np.histogram(subset, bins=bin_count, range=y_limits)
        
        if counts.max() == 0: continue

        # 2. Normalize counts to fit within the designated width
        # The longest bar will be exactly 'max_bar_width' long
        normalized_widths = (counts / counts.max()) * max_bar_width
        
        # 3. Draw Horizontal Bars (barh)
        # left: The starting x-coordinate of the bar (the flat back of the histogram)
        # width: How far to the right it extends
        heights = np.diff(bin_edges)
        bottoms = bin_edges[:-1]
        
        # "One-Sided": All bars start at the same X coordinate
        left_positions = layer + start_offset
        
        ax.barh(y=bottoms, width=normalized_widths, height=heights, 
                left=left_positions, color=color, edgecolor=color, linewidth=0.5, alpha=0.9)

        # 4. Add Mean Marker
        mean_val = np.mean(subset)
        
        # Draw a line across the width of this specific histogram
        # From the start (flat back) to the end (max_bar_width)
        ax.hlines(mean_val, layer + start_offset, layer + start_offset + max_bar_width, 
                  color=MEAN_LINE_COLOR, linewidth=1.5, linestyle='-', zorder=5)
        
        # 5. Add Mean Text Label
        # Place it just to the right of the histogram block for readability
        x_right = layer + start_offset     # right end of the mean line
        y_text  = mean_val + (y_limits[1] - y_limits[0]) * 0.01  # ~1% of y-range above the line

        ax.text(
            x_right, y_text, f"{mean_val:.2f}",
            ha='left', va='bottom', fontsize=8,
            color=MEAN_LINE_COLOR, fontweight='bold'
        )

    # Formatting
    # ax.set_title(title, fontsize=14, pad=10)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_xticks(layers)
    ax.set_ylim(y_limits)
    
    # Grid and Spines
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.grid(axis='x', alpha=0.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def main(args):
    # 1. Load Data
    df = load_data_into_dataframe(args)
    
    if df.empty:
        print("No data found or computed.")
        return

    # 2. Setup Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    # 3. Plot Cosine Similarity
    plot_one_sided_histograms(
        ax=ax1, 
        df=df, 
        metric_col='cosine', 
        color=HIST_COLOR_COS, 
        title="Gradient Coherence: Cosine Similarity",
        y_label="Cosine Similarity",
        y_limits=(-1.05, 1.05)
    )

    # 4. Plot Angle
    plot_one_sided_histograms(
        ax=ax2, 
        df=df, 
        metric_col='angle', 
        color=HIST_COLOR_ANG, 
        title="Gradient Coherence: Angle Distribution",
        y_label="Angle (Degrees)",
        y_limits=(0, 185)
    )

    plt.tight_layout()
    plot_path = os.path.join(args.input_dir, "gradient_coherence_condensed.png")
    plt.savefig(plot_path, dpi=200)
    print(f"\nSaved condensed analysis to: {plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Target Gradient Coherence (One-Sided)")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Root output directory containing the 'target_gradients' folder")
    parser.add_argument("--target_layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    
    args = parser.parse_args()
    main(args)