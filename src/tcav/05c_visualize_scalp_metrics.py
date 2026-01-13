import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import mne
from mne.viz.topomap import _find_topomap_coords
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap


# --- 1. Standard Channel List (Must match your data order) ---
STANDARD_CHANNELS = [
    'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'
]

def load_aggregated_metrics(args):
    """
    Scans the concept directories and aggregates metrics into (Layers x Channels) matrices.
    Returns:
        means_grid: (12, 64) - Mean value of the metric
        sig_grid:   (12, 64) - Boolean (Is Significant based on CSV)
        effect_grid:(12, 64) - Cohen's d
    """
    n_layers = 12
    n_chans = len(STANDARD_CHANNELS)
    
    means_grid = np.full((n_layers, n_chans), np.nan)
    sig_grid = np.zeros((n_layers, n_chans), dtype=bool)
    effect_grid = np.zeros((n_layers, n_chans))
    
    print(f"Aggregating {args.metric} for {args.band} / {args.level} ({args.cav_type} CAV)...")
    
    for ch_idx, channel in enumerate(tqdm(STANDARD_CHANNELS, desc="Loading Channels")):
        # Construct the concept name expected in the directory structure
        # Pattern: {base}/concepts/{band}_{channel}_{level}/3_real_vs_null_cavs/...
        concept_name = f"{args.band}_{channel}_{args.level}"
        concept_dir = os.path.join(args.base_dir, "concepts", concept_name)
        stats_dir = os.path.join(concept_dir, "3_real_vs_null_cavs")
        
        # Filename pattern from 05b: stats_summary_{concept}_{cav_type}.csv
        csv_name = f"stats_summary_concept_set_{args.cav_type}.csv"
        csv_path = os.path.join(stats_dir, csv_name)
        
        if not os.path.exists(csv_path):
            # If a channel is missing, we leave it as NaN/False
            continue
            
        try:
            df = pd.read_csv(csv_path)
            
            # Filter for the specific metric we want (e.g., "Accuracy", "AUC")
            # The CSV has a 'Metric' column
            df_metric = df[df['Metric'] == args.metric]
            
            for _, row in df_metric.iterrows():
                layer = int(row['Layer'])
                if layer < n_layers:
                    means_grid[layer, ch_idx] = row['Real_Mean']
                    sig_grid[layer, ch_idx] = bool(row['Is_Significant'])
                    effect_grid[layer, ch_idx] = row['Effect_Size_Cohen_d']
                    
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            
    return means_grid, sig_grid, effect_grid

def get_channel_coords():
    """Returns 2D coordinates for the standard channel list using MNE."""
    montage = mne.channels.make_standard_montage('biosemi64')
    # Filter montage to only our channels if necessary, or create info with them
    # Note: If STANDARD_CHANNELS contains non-standard names, MNE might error.
    # We assume standard 10-20/10-10 names.
    
    try:
        info = mne.create_info(ch_names=STANDARD_CHANNELS, sfreq=1, ch_types='eeg')
        info.set_montage(montage)
        
        # Project 3D positions to 2D
        pos_3d = np.array([montage.get_positions()['ch_pos'][ch] for ch in STANDARD_CHANNELS])
        max_dist = np.max(np.linalg.norm(pos_3d, axis=1))
        
        # Create a sphere slightly larger than the head for projection
        custom_sphere = (0.0, 0.0, 0.0, max_dist)
        picks = range(len(STANDARD_CHANNELS))
        pos_2d = _find_topomap_coords(info, picks=picks, sphere=custom_sphere)
        
        return pos_2d, max_dist
        
    except ValueError as e:
        print(f"Error creating montage: {e}")
        print("Falling back to random coordinates (DEBUG ONLY). Check channel names.")
        return np.random.rand(len(STANDARD_CHANNELS), 2), 1.0

def plot_scalp_grid(means, sigs, effects, coords, max_dist, args):
    """
    Plots the 4x3 grid of scalp maps.
    """
    print(f"Generating Visualization (Cutoff: d > {args.min_effect_size})...")
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    # --- 1. Determine Color Scale ---
    # We want the colorbar to be symmetric or logical based on the metric
    if args.metric == "Consistency":
        # Cosine similarity: 0 to 1, usually high
        vmin, vmax = 0.5, 1.0
        cmap = plt.get_cmap('Reds')
    elif args.metric in ["Accuracy", "AUC"]:
        # 0.5 to 1.0
        vmin, vmax = 0.5, 1.0 #0.8? Adjust based on data range
        
        # Auto-adjust max if data is super strong
        data_max = np.nanmax(means)
        if data_max > 0.9: 
            vmax = 1.0
        else: 
            # vmax = min(data_max + 0.05, 0.8)
            vmax = data_max + 0.05
        
        # cmap = plt.get_cmap('plasma') 
        cmap = LinearSegmentedColormap.from_list(
                                                "peach_to_red",
                                                ["#FFCB8D", "#D72638"]
                                            )
    else:
        vmin, vmax = np.nanmin(means), np.nanmax(means)
        cmap = plt.get_cmap('viridis')

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # --- 2. Iterate Layers ---
    for layer_id in range(12):
        ax = axes[layer_id]
        
        # Get data for this layer
        l_means = means[layer_id]
        l_sigs = sigs[layer_id]
        l_effects = effects[layer_id]
        
        # Draw Head Outline
        head_circle = patches.Circle((0, 0), radius=max_dist, color='black', fill=False, linewidth=1, alpha=0.3)
        ax.add_patch(head_circle)
        
        # Nose (Optional schematic)
        nose_x = [0, max_dist*0.1, -max_dist*0.1, 0]
        nose_y = [max_dist*1.1, max_dist*1.0, max_dist*1.0, max_dist*1.1]
        ax.plot(nose_x, nose_y, color='black', linewidth=1, alpha=0.3)

        # --- 3. Plot Channels ---
        # We classify channels into two groups: "Show" and "Hide"
        # Show = Significant AND High Effect
        # Hide = Not Significant OR Low Effect
        
        is_shown = (l_sigs == True) & (np.abs(l_effects) >= args.min_effect_size)
        
        # A. Plot Hidden (Insignificant) - Small Dots
        hidden_idx = np.where(~is_shown)[0]
        if len(hidden_idx) > 0:
            ax.scatter(coords[hidden_idx, 0], coords[hidden_idx, 1],
                       c='gray', s=10, alpha=0.3, edgecolors='none', zorder=1)
            
        # B. Plot Shown (Significant) - Large Colored Circles
        shown_idx = np.where(is_shown)[0]
        if len(shown_idx) > 0:
            sc = ax.scatter(coords[shown_idx, 0], coords[shown_idx, 1],
                            c=l_means[shown_idx], cmap=cmap, norm=norm,
                            s=450, edgecolors='grey', linewidths=0.1, zorder=2)
            
            # Add Labels
            for idx in shown_idx:
                ax.text(coords[idx, 0], coords[idx, 1], STANDARD_CHANNELS[idx],
                        ha='center', va='center', fontsize=8, color='white', 
                        fontweight='bold', zorder=3)
        
        # Formatting
        ax.set_title(f"Layer {layer_id}", fontsize=12)
        ax.set_aspect('equal')
        ax.axis('off')
        lim = max_dist * 1.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    # --- 4. Colorbar & Layout ---
    # Leave room on the right for the colorbar: [left, bottom, right, top]
    fig.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])

    # Add colorbar on the right (in the reserved space)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"Mean {args.metric}", fontsize=14)
    
    # # Title
    # fig.suptitle(f"Scalp Distribution: {args.band} / {args.level}\n"
    #              f"Metric: {args.metric} ({args.cav_type} CAV) | Cutoff: p<0.05 & |d|>{args.min_effect_size}", 
    #              fontsize=16, y=1.02)
    
    # Save
    out_name = f"scalp_viz_{args.band}_{args.level}_{args.cav_type}_{args.metric}.png"
    out_path = os.path.join(args.output_dir, out_name)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved visualization to: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize aggregated CAV metrics across the scalp.")
    
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Root directory containing the 'concepts' folder.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save the resulting plot.")
    
    # Selection parameters
    parser.add_argument("--band", type=str, required=True, choices=['alpha', 'beta'],
                        help="Frequency band (e.g., alpha).")
    parser.add_argument("--level", type=str, required=True, choices=['high', 'low'],
                        help="Level (e.g., high).")
    parser.add_argument("--cav_type", type=str, default="filter", choices=['filter', 'pattern'],
                        help="Type of CAV (filter or pattern).")
    parser.add_argument("--metric", type=str, default="Accuracy", choices=['Accuracy', 'AUC', 'Consistency'],
                        help="Metric to visualize.")
    
    # Filtering parameters
    parser.add_argument("--min_effect_size", type=float, default=0.8,
                        help="Minimum Cohen's d (absolute value) to show a channel.")
    
    args = parser.parse_args()
    
    # 1. Prepare
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Load Data
    means, sigs, effects = load_aggregated_metrics(args)
    
    # Check if we have data
    if np.all(np.isnan(means)):
        print("No data found! Check input paths and verify that 05b ran successfully.")
        return

    # 3. Get Coordinates
    coords, max_dist = get_channel_coords()
    
    # 4. Plot
    plot_scalp_grid(means, sigs, effects, coords, max_dist, args)

if __name__ == "__main__":
    main()