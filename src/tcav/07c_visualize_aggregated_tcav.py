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

# --- 1. Standard Channel List ---
STANDARD_CHANNELS = [
    'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'
]

def load_csv_data(base_dir, band, level, cav_type, metric_name, step_folder):
    """
    Generic function to crawl CSVs and return data grids.
    step_folder: '3_real_vs_null_cavs' (for Accuracy) or '5_real_vs_null_tcav_scores' (for TCAV)
    """
    n_layers = 12
    n_chans = len(STANDARD_CHANNELS)
    
    means_grid = np.full((n_layers, n_chans), np.nan)
    sig_grid = np.zeros((n_layers, n_chans), dtype=bool)
    effect_grid = np.zeros((n_layers, n_chans))
    
    # We construct the CSV filename differently based on the step
    # 05b (CAV) -> stats_summary_{concept}_{cav_type}.csv
    # 07b (TCAV) -> stats_summary_{concept}_{cav_type}.csv (Same format)
    
    print(f"  Loading {metric_name} from {step_folder}...")
    
    for ch_idx, channel in enumerate(STANDARD_CHANNELS):
        concept_name = f"{band}_{channel}_{level}"
        csv_path = os.path.join(
            base_dir, "concepts", concept_name, step_folder, 
            f"stats_summary_concept_set_{cav_type}.csv"
        )
        
        if not os.path.exists(csv_path):
            continue
            
        try:
            df = pd.read_csv(csv_path)
            # Filter for the specific metric
            df_metric = df[df['Metric'] == metric_name]
            
            for _, row in df_metric.iterrows():
                layer = int(row['Layer'])
                if layer < n_layers:
                    means_grid[layer, ch_idx] = row['Real_Mean']
                    sig_grid[layer, ch_idx] = bool(row['Is_Significant'])
                    effect_grid[layer, ch_idx] = row['Effect_Size_Cohen_d']
                    
        except Exception:
            pass # Skip broken files silently
            
    return means_grid, sig_grid, effect_grid

def get_channel_coords():
    """Returns 2D coordinates for the standard channel list."""
    montage = mne.channels.make_standard_montage('biosemi64')
    try:
        info = mne.create_info(ch_names=STANDARD_CHANNELS, sfreq=1, ch_types='eeg')
        info.set_montage(montage)
        pos_3d = np.array([montage.get_positions()['ch_pos'][ch] for ch in STANDARD_CHANNELS])
        max_dist = np.max(np.linalg.norm(pos_3d, axis=1))
        custom_sphere = (0.0, 0.0, 0.0, max_dist)
        picks = range(len(STANDARD_CHANNELS))
        pos_2d = _find_topomap_coords(info, picks=picks, sphere=custom_sphere)
        return pos_2d, max_dist
    except Exception as e:
        print(f"Error creating montage: {e}")
        return np.random.rand(len(STANDARD_CHANNELS), 2), 1.0

def plot_scalp_grid(tcav_means, tcav_sigs, tcav_effects, 
                    acc_means, 
                    coords, max_dist, args):
    """
    Plots the 4x3 grid with Dual Filtering.
    """
    print(f"Generating Visualization...")
    print(f"  Filters: TCAV |d| > {args.min_effect_size} AND CAV Acc > {args.min_accuracy}")

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    # --- 1. Color Scale Setup ---
    if args.metric == "TCAV Score":
        # Diverging centered at 0.5
        # Blue (0.0) -> White (0.5) -> Red (1.0)
        vmin, vmax = 0.0, 1.0
        # Custom diverging colormap to ensure 0.5 is perfectly white
        colors = ["#03045E", "#ffffff", "#D72638"] # Blue -> White -> Red
        cmap = LinearSegmentedColormap.from_list("custom_bwr", colors, N=256)
        # We can also use 'bwr' or 'coolwarm'
    elif args.metric in ["Cosine Similarity", "Sensitivity"]:
        # Diverging centered at 0.0
        # Purple/Pink (Negative) -> White (0) -> Green (Positive)
        limit = max(abs(np.nanmin(tcav_means)), abs(np.nanmax(tcav_means)))
        if limit == 0: limit = 0.1
        vmin, vmax = -limit, limit
        colors = ["#03045E", "#ffffff", "#D72638"] # Blue -> White -> Red
        cmap = LinearSegmentedColormap.from_list("custom_bwr", colors, N=256)
    else:
        vmin, vmax = np.nanmin(tcav_means), np.nanmax(tcav_means)
        colors = ["#03045E", "#ffffff", "#D72638"] # Blue -> White -> Red
        cmap = LinearSegmentedColormap.from_list("custom_bwr", colors, N=256)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # --- 2. Iterate Layers ---
    for layer_id in range(12):
        ax = axes[layer_id]
        
        # Get data vectors for this layer
        t_means = tcav_means[layer_id]
        t_sigs  = tcav_sigs[layer_id]
        t_effs  = tcav_effects[layer_id]
        a_means = acc_means[layer_id]
        
        # Draw Head
        head_circle = patches.Circle((0, 0), radius=max_dist, color='black', fill=False, linewidth=1, alpha=0.3)
        ax.add_patch(head_circle)
        nose_x = [0, max_dist*0.1, -max_dist*0.1, 0]
        nose_y = [max_dist*1.1, max_dist*1.0, max_dist*1.0, max_dist*1.1]
        ax.plot(nose_x, nose_y, color='black', linewidth=1, alpha=0.3)

        # --- 3. Apply Dual Filters ---
        # Condition A: TCAV Metric is statistically valid
        valid_tcav = (t_sigs == True) & (np.abs(t_effs) >= args.min_effect_size)
        
        # Condition B: Concept was learned accurately (Gating)
        # Note: We handle NaNs in accuracy (treat as 0.5/Fail)
        safe_acc = np.nan_to_num(a_means, nan=0.5)
        valid_concept = safe_acc >= args.min_accuracy
        
        # Final Mask
        is_shown = valid_tcav & valid_concept
        
        # --- 4. Plot ---
        # A. Hidden (Tiny Dots)
        hidden_idx = np.where(~is_shown)[0]
        if len(hidden_idx) > 0:
            ax.scatter(coords[hidden_idx, 0], coords[hidden_idx, 1],
                       c='gray', s=10, alpha=0.3, edgecolors='none', zorder=1)
            
        # B. Shown (Large Colored Circles)
        shown_idx = np.where(is_shown)[0]
        if len(shown_idx) > 0:
            sc = ax.scatter(coords[shown_idx, 0], coords[shown_idx, 1],
                            c=t_means[shown_idx], cmap=cmap, norm=norm,
                            s=450, edgecolors='grey', linewidths=0.1, zorder=2)
            
            # Labels
            for idx in shown_idx:
                # Text color logic: White for dark bubbles, Black for light bubbles
                val = t_means[idx]
                if args.metric == "TCAV Score":
                    txt_col = 'white' if (val < 0.3 or val > 0.7) else 'black'
                    fontweight='bold' if txt_col == 'white' else 'normal'

                elif args.metric in ["Cosine Similarity", "Sensitivity"]:
                    mid_point = 0.0
                    txt_col = 'white' if (val < mid_point - (vmax - vmin)/6 or val > mid_point + (vmax - vmin)/6) else 'black'
                    fontweight='bold' if txt_col == 'white' else 'normal'
                else:
                    txt_col = 'black'
                    
                ax.text(coords[idx, 0], coords[idx, 1], STANDARD_CHANNELS[idx],
                        ha='center', va='center', fontsize=8, color=txt_col, 
                        fontweight=fontweight, zorder=3)
        
        ax.set_title(f"Layer {layer_id}", fontsize=12)
        ax.set_aspect('equal')
        ax.axis('off')
        lim = max_dist * 1.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    # --- 5. Legends & Colorbar ---
    fig.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"Mean {args.metric}", fontsize=14)
    
    # Add Tick Labels for TCAV Score context
    if args.metric == "TCAV Score":
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(['Negative (0.0)', 'Neutral (0.5)', 'Positive (1.0)'])

    # fig.suptitle(f"Scalp Distribution: {args.band} / {args.level}\n"
    #              f"Metric: {args.metric} ({args.cav_type}) | "
    #              f"Filters: p<0.05, |d|>{args.min_effect_size}, Acc>{args.min_accuracy*100:.0f}%", 
    #              fontsize=16, y=1.02)
    
    # Save
    safe_metric = args.metric.replace(" ", "_")
    out_name = f"scalp_tcav_{args.band}_{args.level}_{args.cav_type}_{safe_metric}.png"
    out_path = os.path.join(args.output_dir, out_name)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved visualization to: {out_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize aggregated TCAV metrics with Accuracy Gating.")
    
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    # Selection
    parser.add_argument("--band", type=str, required=True)
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--cav_type", type=str, default="filter")
    parser.add_argument("--metric", type=str, default="TCAV Score", 
                        choices=['TCAV Score', 'Cosine Similarity', 'Sensitivity'])
    
    # Thresholds
    parser.add_argument("--min_effect_size", type=float, default=0.8,
                        help="Min Cohen's d for the TCAV metric itself.")
    parser.add_argument("--min_accuracy", type=float, default=0.60,
                        help="Min CAV Accuracy required to display the channel.")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load TCAV Stats (07b output)
    print("Loading TCAV Metrics...")
    tcav_means, tcav_sigs, tcav_effs = load_csv_data(
        args.base_dir, args.band, args.level, args.cav_type, 
        args.metric, "5_real_vs_null_tcav_scores"
    )
    
    # 2. Load Accuracy Stats (05b output) for Gating
    print("Loading Accuracy Metrics (for Gating)...")
    acc_means, _, _ = load_csv_data(
        args.base_dir, args.band, args.level, args.cav_type, 
        "Accuracy", "3_real_vs_null_cavs"
    )
    
    if np.all(np.isnan(tcav_means)):
        print("No TCAV data found. Check inputs.")
        return

    # 3. Plot
    coords, max_dist = get_channel_coords()
    plot_scalp_grid(tcav_means, tcav_sigs, tcav_effs, acc_means, coords, max_dist, args)

if __name__ == "__main__":
    main()