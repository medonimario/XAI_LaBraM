import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.patches as patches

# Set seaborn style for plots
sns.set_theme(style="whitegrid")

def load_relative_tcav_data(json_path):
    """
    Loads Relative TCAV results.
    Returns a single DataFrame containing Real and Null values for ALL metrics.
    Columns: [layer_id, type, metric, dist, value]
    """
    print(f"Loading data from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load file. {e}")
        return None

    rows = []
    
    for layer_id_str, layer_data in data.items():
        layer_id = int(layer_id_str)
        real = layer_data['real']
        null = layer_data['null']

        # Helper to add data rows
        def add_metric(cav_type, metric_name, real_val, null_vals):
            # Add Null Distribution
            for v in null_vals:
                rows.append({
                    'layer_id': layer_id,
                    'type': cav_type,
                    'metric': metric_name,
                    'dist': 'null',
                    'value': v
                })
            # Add Real Value
            rows.append({
                'layer_id': layer_id,
                'type': cav_type,
                'metric': metric_name,
                'dist': 'real',
                'value': real_val
            })

        # --- 1. Filter CAV Metrics ---
        # Cosine Similarity
        add_metric('filter', 'cosine_sim', 
                   real['filter']['cosine_sim'], 
                   null['filter_sims'])
        # Accuracy (Quality)
        add_metric('filter', 'quality', 
                   real['filter']['accuracy'], 
                   null['filter_accs'])
        # TCAV Score
        add_metric('filter', 'tcav_score', 
                   real['filter']['tcav'], 
                   null['filter_tcavs'])

        # --- 2. Pattern CAV Metrics ---
        # Cosine Similarity
        add_metric('pattern', 'cosine_sim', 
                   real['pattern']['cosine_sim'], 
                   null['pattern_sims'])
        # AUC (Quality)
        add_metric('pattern', 'quality', 
                   real['pattern']['auc'], 
                   null['pattern_aucs'])
        # TCAV Score
        add_metric('pattern', 'tcav_score', 
                   real['pattern']['tcav'], 
                   null['pattern_tcavs'])

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} data points across {len(df['layer_id'].unique())} layers.")
    return df

def calculate_empirical_stats(df, alpha=0.05):
    """
    Calculates Empirical P-Values for Real vs Null distributions.
    Applies FDR correction per (cav_type, metric) group.
    """
    stats_results = []
    
    # Get unique combinations of (type, metric) to correct separately
    groups = df[['type', 'metric']].drop_duplicates().values
    
    print("Calculating Empirical P-Values and applying FDR correction...")

    for cav_type, metric in groups:
        group_p_values = []
        group_indices = [] # To map back after correction
        
        layers = sorted(df['layer_id'].unique())
        
        for layer in layers:
            subset = df[(df['layer_id'] == layer) & 
                        (df['type'] == cav_type) & 
                        (df['metric'] == metric)]
            
            real_data = subset[subset['dist'] == 'real']['value'].values
            null_data = subset[subset['dist'] == 'null']['value'].values
            
            if len(real_data) == 0 or len(null_data) == 0:
                continue
                
            real_val = real_data[0]
            
            # One-sided test: Is Real > Null?
            # p = (count(null >= real) + 1) / (N + 1)
            n_extreme = np.sum(null_data >= real_val)
            n_perm = len(null_data)
            p = (n_extreme + 1) / (n_perm + 1)
            
            group_p_values.append(p)
            
            stats_results.append({
                'layer_id': layer,
                'type': cav_type,
                'metric': metric,
                'real_value': real_val,
                'null_mean': np.mean(null_data),
                'null_std': np.std(null_data),
                'p_uncorrected': p
            })
            
        # Apply FDR Correction for this group
        if group_p_values:
            rejected, p_corrected = fdrcorrection(group_p_values, alpha=alpha, method='indep')
            
            # Update the dictionaries in stats_results
            # We iterate backwards or match by layer/type/metric
            count = 0
            for res in stats_results:
                if res['type'] == cav_type and res['metric'] == metric:
                    res['p_corrected'] = p_corrected[count]
                    res['significant'] = rejected[count]
                    count += 1

    return pd.DataFrame(stats_results)

def plot_raincloud_metric(df, stats_df, cav_type, metric, title, ylabel, output_path, palette):
    """
    Plots a Raincloud (Half Violin Null + Star Real) for a specific metric.
    """
    # Filter data
    plot_data = df[(df['type'] == cav_type) & (df['metric'] == metric)]
    layers = sorted(plot_data['layer_id'].unique())
    
    if plot_data.empty:
        print(f"No data for {cav_type} - {metric}, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, layer in enumerate(layers):
        layer_df = plot_data[plot_data['layer_id'] == layer]
        null_vals = layer_df[layer_df['dist'] == 'null']['value'].values
        real_vals = layer_df[layer_df['dist'] == 'real']['value'].values
        
        if len(real_vals) == 0: continue
        real_val = real_vals[0]
        color = palette[layer]
        
        # --- A. Null Distribution (Left Half Violin) ---
        if len(null_vals) > 0:
            parts = ax.violinplot(null_vals, positions=[i], widths=0.7,
                                  showmeans=False, showmedians=False, showextrema=False)
            for body in parts['bodies']:
                # Clip path to left side
                m = np.mean(body.get_paths()[0].vertices[:, 0])
                body.get_paths()[0].vertices[:, 0] = np.clip(body.get_paths()[0].vertices[:, 0], -np.inf, m)
                body.set_facecolor('gray')
                body.set_alpha(0.3)
                body.set_edgecolor('none')
            
            # Add mean marker for null
            ax.plot(i - 0.1, np.mean(null_vals), 'o', color='gray', alpha=0.6, markersize=4)

        # --- B. Real Value (Right Side Star) ---
        ax.scatter(i + 0.1, real_val, marker='*', s=250, c=[color], edgecolor='black', zorder=10, label='Real' if i==0 else "")
        
        # --- C. Significance Star ---
        # Check stats
        row = stats_df[(stats_df['layer_id'] == layer) & 
                       (stats_df['type'] == cav_type) & 
                       (stats_df['metric'] == metric)]
        
        if not row.empty and row.iloc[0]['significant']:
            # Place asterisk slightly above the real value or the top of the violin
            y_pos = max(real_val, np.max(null_vals)) if len(null_vals) > 0 else real_val
            ax.text(i, y_pos + (y_pos*0.05), '*', ha='center', va='bottom', fontsize=20, fontweight='bold', color='black')

    # Formatting
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=12)
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    
    # Reference Line (0.5 for Accuracy/AUC/TCAV, 0 for Cosine)
    if 'cosine' in metric:
        ax.axhline(0, ls='--', color='black', alpha=0.3)
    else:
        ax.axhline(0.5, ls='--', color='black', alpha=0.3, label='Chance/Neutral')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', lw=4, alpha=0.3, label='Null Dist (Permuted)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=15, label='Real Concept'),
        Line2D([0], [0], marker='$*$', color='w', markerfacecolor='black', markersize=15, label='Significant (p<0.05)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Generated plot: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    df = load_relative_tcav_data(args.input_json)
    if df is None: return

    # 2. Statistics
    stats_df = calculate_empirical_stats(df, args.alpha)
    stats_path = os.path.join(args.output_dir, "relative_tcav_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved statistics summary to {stats_path}")
    
    # 3. Generate All Plots
    layers = sorted(df['layer_id'].unique())
    plasma_colors = sns.color_palette("plasma", n_colors=len(layers))
    palette = dict(zip(layers, plasma_colors))
    
    # Define plot configurations
    # (Type, Metric, Title, YLabel, Filename)
    plots = [
        ('filter', 'quality', 'Filter CAV: Accuracy (Real vs Null)', 'Accuracy', 'plot_filter_accuracy.png'),
        ('filter', 'cosine_sim', 'Filter CAV: Cosine Similarity (Real vs Null)', 'Cosine Similarity', 'plot_filter_similarity.png'),
        ('filter', 'tcav_score', 'Filter CAV: TCAV Score (Real vs Null)', 'TCAV Score', 'plot_filter_tcav.png'),
        
        ('pattern', 'quality', 'Pattern CAV: AUC (Real vs Null)', 'AUC Score', 'plot_pattern_auc.png'),
        ('pattern', 'cosine_sim', 'Pattern CAV: Cosine Similarity (Real vs Null)', 'Cosine Similarity', 'plot_pattern_similarity.png'),
        ('pattern', 'tcav_score', 'Pattern CAV: TCAV Score (Real vs Null)', 'TCAV Score', 'plot_pattern_tcav.png'),
    ]
    
    for c_type, metric, title, ylabel, fname in plots:
        out_path = os.path.join(args.output_dir, fname)
        plot_raincloud_metric(df, stats_df, c_type, metric, title, ylabel, out_path, palette)

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main()