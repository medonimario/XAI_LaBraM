import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.patches as patches # Import for manual boxplot

# Set seaborn style for plots
sns.set_theme(style="whitegrid")

def load_tcav_data(json_path):
    """
    Loads the TCAV statistics JSON and parses it into a DataFrame.
    
    Args:
        json_path (str): Path to the tcav_statistics_all_runs.json file.

    Returns:
        pd.DataFrame: A DataFrame with columns 
                      ['run_id', 'layer_id', 'cav_accuracy', 'tcav_score'].
    """
    print(f"Loading data from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found at {json_path}")
        print("Please run 03_run_tcav.py first to generate this file.")
        return None
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse {json_path}. File might be empty or corrupt.")
        return None

    all_records = []
    for run in data:
        run_id = run.get('run_id')
        for layer_id, scores in run.get('layers', {}).items():
            all_records.append({
                'run_id': run_id,
                'layer_id': int(layer_id), # Ensure layer is int for correct sorting
                'cav_accuracy': scores.get('cav_accuracy'),
                'tcav_score': scores.get('tcav_score')
            })
            
    if not all_records:
        print("ERROR: No data records found in the JSON file.")
        return None

    df = pd.DataFrame(all_records)
    df.dropna(inplace=True) # Drop any NaN values from failed runs
    
    print(f"Successfully loaded {len(df)} records from {len(data)} runs.")
    return df

def perform_statistics(df, alpha=0.05):
    """
    Performs one-sample Wilcoxon signed-rank test against 0.5 for each layer
    and applies FDR correction.

    Args:
        df (pd.DataFrame): The loaded TCAV data.
        alpha (float): Significance level.

    Returns:
        pd.DataFrame: A DataFrame with summary stats and p-values.
    """
    layers = sorted(df['layer_id'].unique())
    stats_results = []

    pvals_cav = []
    pvals_tcav = []
    means_cav = []
    means_tcav = []

    print("Running statistical tests...")
    # --- 1. Get p-values for each layer ---
    for layer in layers:
        layer_data = df[df['layer_id'] == layer]
        
        cav_scores = layer_data['cav_accuracy'].values
        tcav_scores = layer_data['tcav_score'].values
        
        means_cav.append(np.mean(cav_scores))
        means_tcav.append(np.mean(tcav_scores))
        
        # Test if scores are different from 0.5
        # We use scores - 0.5 and test against 0
        try:
            # Add a small epsilon if all values are exactly 0.5 (avoids error)
            cav_diff = cav_scores - 0.5
            tcav_diff = tcav_scores - 0.5
            if np.all(cav_diff == 0): cav_diff[0] += 1e-9
            if np.all(tcav_diff == 0): tcav_diff[0] += 1e-9

            stat_cav, p_cav = wilcoxon(cav_diff, alternative='two-sided')
            stat_tcav, p_tcav = wilcoxon(tcav_diff, alternative='two-sided')
            
            pvals_cav.append(p_cav)
            pvals_tcav.append(p_tcav)
        except ValueError as e:
            print(f"  Warning: Stats error for layer {layer} ({e}). Appending p=1.0")
            pvals_cav.append(1.0)
            pvals_tcav.append(1.0)

    # --- 2. Apply FDR correction ---
    sig_cav, p_cav_corrected = fdrcorrection(pvals_cav, alpha=alpha, method='indep')
    sig_tcav, p_tcav_corrected = fdrcorrection(pvals_tcav, alpha=alpha, method='indep')

    # --- 3. Compile results ---
    for i, layer in enumerate(layers):
        stats_results.append({
            'layer_id': layer,
            'mean_cav_acc': means_cav[i],
            'mean_tcav_score': means_tcav[i],
            'p_cav_uncorrected': pvals_cav[i],
            'p_tcav_uncorrected': pvals_tcav[i],
            'p_cav_corrected': p_cav_corrected[i],
            'p_tcav_corrected': p_tcav_corrected[i],
            'significant_cav': sig_cav[i],
            'significant_tcav': sig_tcav[i],
        })

    stats_df = pd.DataFrame(stats_results)
    
    print("\n--- Statistical Results (FDR-Corrected) ---")
    print(stats_df)
    print("---------------------------------------------")
    return stats_df

def draw_half_raincloud_plot(ax, df, stats_df, score_col, sig_col, palette, baseline_color):
    """
    Helper function to draw a single half-raincloud plot (Rain+Cloud+Box)
    on a given Axes object.
    """
    layers = sorted(df['layer_id'].unique())
    
    for i, layer in enumerate(layers):
        data = df[df['layer_id'] == layer][score_col].dropna().values
        if len(data) == 0:
            continue
        
        # *** MODIFIED: Use the layer index for the palette color ***
        color = palette[layer]
        
        # 1. Draw Scatter ("Rain") (Left Side)
        jitter = np.random.uniform(i - 0.3, i - 0.1, size=len(data))
        ax.scatter(jitter, data, s=15, c=[color], alpha=0.6, zorder=2, label=None)

        # 2. Draw Half-Violin (Right Side)
        v = ax.violinplot(data, positions=[i], widths=0.7, 
                          showmeans=False, showmedians=False, showextrema=False)
        
        # Clip to the right half
        body = v['bodies'][0]
        verts = body.get_paths()[0].vertices
        verts[:, 0] = np.clip(verts[:, 0], i, i + 0.35)
        body.set_facecolor(color)
        body.set_alpha(0.6)
        body.set_edgecolor('none')
        
        # Hide default violin lines
        if 'cbars' in v: v['cbars'].set_visible(False)
        if 'cmins' in v: v['cmins'].set_visible(False)
        if 'cmaxes' in v: v['cmaxes'].set_visible(False)

        # 3. Draw Manual Half-Boxplot (Right Side, as in inspiration)
        q1, median, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        non_outliers = data[(data >= lower_fence) & (data <= upper_fence)]
        if non_outliers.size == 0:
            wlow, whigh = q1, q3
        else:
            wlow, whigh = non_outliers.min(), non_outliers.max()

        x_center = i
        box_width = 0.05
        x_left = x_center + 0.02  # Start slightly right of center
        x_right = x_left + box_width

        # Box
        rect = patches.Rectangle(
            (x_left, q1), box_width, q3 - q1,
            facecolor=color, edgecolor='grey', alpha=0.7,
            linewidth=0.5, zorder=3
        )
        ax.add_patch(rect)
        
        # Median
        ax.plot([x_left, x_right], [median, median], color="white", lw=1.0, zorder=4, solid_capstyle='butt')
        
        # Whiskers
        ax.plot([x_left, x_left], [q3, whigh], color="grey", lw=0.5, zorder=3, solid_capstyle='butt')
        ax.plot([x_left, x_left], [q1, wlow], color="grey", lw=0.5, zorder=3, solid_capstyle='butt')

    # --- Post-loop formatting ---
    
    # 4. Add Random Baseline line
    ax.axhline(0.5, ls='--', color=baseline_color, label='Random Baseline (0.5)')
    
    # 5. Add Significance Stars
    max_val = df[score_col].max()
    star_height = max(1.05, max_val * 1.05)
    ax.set_ylim(bottom=0, top=star_height * 1.05) # Make space
    
    for _, row in stats_df.iterrows():
        if row[sig_col]:
            ax.text(row['layer_id'], star_height, '*', 
                        ha='center', va='bottom', color='black', fontsize=20)

    ax.set_xticks(layers)
    ax.set_xticklabels(layers)
    ax.legend()


def plot_rainclouds(df, stats_df, output_dir):
    """
    Generates and saves half-raincloud plots for CAV and TCAV scores,
    inspired by the user-provided code.
    """
    print("Generating half-raincloud plots...")
    
    layers = sorted(df['layer_id'].unique())
    
    # *** MODIFIED: Use 'plasma' palette for both ***
    # We generate a list of colors from the plasma map
    plasma_colors = sns.color_palette("plasma", n_colors=len(layers))
    # We create a dictionary mapping layer_id -> color
    palette = dict(zip(layers, plasma_colors))

    # --- Plot 1: CAV Classifier Accuracy ---
    fig_cav, ax_cav = plt.subplots(figsize=(16, 8))
    
    draw_half_raincloud_plot(ax_cav, df, stats_df,
                             score_col='cav_accuracy',
                             sig_col='significant_cav',
                             palette=palette,  # Pass the plasma palette
                             baseline_color='black')
    
    ax_cav.set_title('Distribution of CAV Classifier Accuracies Across Layers (50 Runs)', fontsize=16)
    ax_cav.set_xlabel('Transformer Block (Layer)', fontsize=12)
    ax_cav.set_ylabel('CAV Classifier Accuracy', fontsize=12)
    # ax_cav.set_ylim(0, 1.05)  # Accuracies are between 0 and 1
    
    plot_path_cav = os.path.join(output_dir, "cav_accuracy_raincloud_plot.png")
    fig_cav.savefig(plot_path_cav, dpi=150)
    print(f"Saved CAV accuracy raincloud plot to {plot_path_cav}")
    plt.close(fig_cav)

    
    # --- Plot 2: TCAV Scores ---
    fig_tcav, ax_tcav = plt.subplots(figsize=(16, 8))
    
    draw_half_raincloud_plot(ax_tcav, df, stats_df,
                             score_col='tcav_score',
                             sig_col='significant_tcav',
                             palette=palette,  # Pass the same plasma palette
                             baseline_color='red')
    
    ax_tcav.set_title('Distribution of TCAV Scores Across Layers (50 Runs)', fontsize=16)
    ax_tcav.set_xlabel('Transformer Block (Layer)', fontsize=12)
    ax_tcav.set_ylabel('TCAV Score', fontsize=12)
    # ax_tcav.set_ylim(0, 1.05)  # TCAV scores are between 0 and 1
    
    plot_path_tcav = os.path.join(output_dir, "tcav_score_raincloud_plot.png")
    fig_tcav.savefig(plot_path_tcav, dpi=150)
    print(f"Saved TCAV score raincloud plot to {plot_path_tcav}")
    plt.close(fig_tcav)


def plot_barplots(stats_df, output_dir):
    """
    Generates and saves bar plots for mean CAV and TCAV scores.
    """
    print("Generating bar plots...")
    
    # *** MODIFIED: Use 'plasma' palette for both ***
    palette = sns.color_palette("plasma", n_colors=len(stats_df))
    
    # --- Plot 1: Mean CAV Accuracy ---
    fig_cav, ax_cav = plt.subplots(figsize=(16, 8))
    
    sns.barplot(x='layer_id', y='mean_cav_acc', data=stats_df, ax=ax_cav, palette=palette)
    
    ax_cav.axhline(0.5, ls='--', color='black', label='Random Chance (0.5)')
    
    # Add Significance Stars
    y_min, y_max = ax_cav.get_ylim()
    star_height = y_max + (y_max - y_min) * 0.02 # 2% above max
    ax_cav.set_ylim(0, star_height * 1.05) # Make space
    
    for _, row in stats_df.iterrows():
        if row['significant_cav']:
            ax_cav.text(row['layer_id'], row['mean_cav_acc'] + 0.01, '*', 
                        ha='center', va='bottom', color='black', fontsize=20)

    ax_cav.set_title('Mean CAV Classifier Accuracy Across Layers', fontsize=16)
    ax_cav.set_xlabel('Transformer Block (Layer)', fontsize=12)
    ax_cav.set_ylabel('Mean CAV Accuracy', fontsize=12)
    ax_cav.set_ylim(0, 1.05)  # Accuracies are between 0 and 1
    ax_cav.legend()

    plot_path_cav = os.path.join(output_dir, "cav_accuracy_barplot.png")
    fig_cav.savefig(plot_path_cav, dpi=150)
    print(f"Saved CAV accuracy bar plot to {plot_path_cav}")
    plt.close(fig_cav)

    
    # --- Plot 2: Mean TCAV Score ---
    fig_tcav, ax_tcav = plt.subplots(figsize=(16, 8))
    
    sns.barplot(x='layer_id', y='mean_tcav_score', data=stats_df, ax=ax_tcav, palette=palette)
    
    ax_tcav.axhline(0.5, ls='--', color='red', label='Random Baseline (0.5)')

    # Add Significance Stars
    y_min, y_max = ax_tcav.get_ylim()
    star_height = y_max + (y_max - y_min) * 0.02 # 2% above max
    ax_tcav.set_ylim(0, star_height * 1.05) # Make space
    
    for _, row in stats_df.iterrows():
        if row['significant_tcav']:
            # Place star just above the bar
            ax_tcav.text(row['layer_id'], row['mean_tcav_score'] + 0.01, '*', 
                        ha='center', va='bottom', color='black', fontsize=20)

    ax_tcav.set_title('Mean TCAV Scores Across Layers', fontsize=16)
    ax_tcav.set_xlabel('Transformer Block (Layer)', fontsize=12)
    ax_tcav.set_ylabel('Mean TCAV Score', fontsize=12)
    ax_tcav.set_ylim(0, 1.05)  # TCAV scores are between 0 and 1
    ax_tcav.legend()

    plot_path_tcav = os.path.join(output_dir, "tcav_score_barplot.png")
    fig_tcav.savefig(plot_path_tcav, dpi=150)
    print(f"Saved TCAV score bar plot to {plot_path_tcav}")
    plt.close(fig_tcav)


def main():
    parser = argparse.ArgumentParser(description="Analyze TCAV results and generate plots.")
    
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to the 'tcav_statistics_all_runs.json' file.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the output plots and stats summary.")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for FDR correction.")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    tcav_df = load_tcav_data(args.input_json)
    if tcav_df is None:
        return
        
    # 2. Perform Statistics
    stats_df = perform_statistics(tcav_df, args.alpha)
    
    # Save stats to a CSV file
    stats_summary_path = os.path.join(args.output_dir, "stats_summary_fdr.csv")
    stats_df.to_csv(stats_summary_path, index=False)
    print(f"\nSaved stats summary to {stats_summary_path}")
    
    # 3. Generate Plots
    plot_rainclouds(tcav_df, stats_df, args.output_dir)
    plot_barplots(stats_df, args.output_dir)
    
    print("\n--- Analysis and plotting complete! ---")

if __name__ == "__main__":
    main()