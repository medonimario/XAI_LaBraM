import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import argparse
from tqdm import tqdm

# --- CONFIGURATION ---
COLORS = {
    'Filter': '#A3D5FF',  # Bright Blue
    'Pattern': '#FF7886'  # Soft Red
}

def compute_cohens_d(x, y):
    """Computes Cohen's d (Effect Size) for two independent samples."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pool_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
    if pool_std == 0: return 0
    return (np.mean(x) - np.mean(y)) / pool_std

def load_data(args):
    """
    Loads final_metrics_summary_{concept}.json and flattens it into a DataFrame.
    """
    json_path = os.path.join(args.input_dir, f"final_metrics_summary_{args.concept_name}.json")
    
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        return pd.DataFrame()

    with open(json_path, 'r') as f:
        data = json.load(f)

    records = []
    # Sort layers numerically
    layer_ids = sorted([int(k) for k in data.keys()])
    
    print("Loading data...")
    for layer_id in layer_ids:
        layer_data = data[str(layer_id)]
        
        for method in ['filter', 'pattern']:
            method_label = 'Filter' if method == 'filter' else 'Pattern'
            
            if method not in layer_data:
                continue
                
            metrics = layer_data[method]
            
            # Extract lists
            scores = metrics.get('tcav_scores', [])
            sensitivities = metrics.get('mean_sensitivities', [])
            cosines = metrics.get('mean_cosines', [])
            
            # Append to records
            for v in scores:
                records.append({'layer': layer_id, 'method': method_label, 'metric': 'TCAV Score', 'value': v})
            for v in sensitivities:
                records.append({'layer': layer_id, 'method': method_label, 'metric': 'Sensitivity', 'value': v})
            for v in cosines:
                records.append({'layer': layer_id, 'method': method_label, 'metric': 'Cosine Similarity', 'value': v})

    return pd.DataFrame(records)

def run_statistics(df, metric_name, group_col='method', groups=['Filter', 'Pattern'], alpha=0.05):
    """
    Runs Mann-Whitney U test per layer, calculates Cohen's d, 
    and applies FDR (Benjamini-Hochberg) correction.
    """
    # Filter for the specific metric first
    df_m = df[df['metric'] == metric_name]
    layers = sorted(df_m['layer'].unique())
    
    p_values = []
    effect_sizes = []
    valid_layers = []

    # 1. Compute Raw Stats
    for l in layers:
        layer_data = df_m[df_m['layer'] == l]
        
        # Extract the two distributions
        data_a = layer_data[layer_data[group_col] == groups[0]]['value'].values
        data_b = layer_data[layer_data[group_col] == groups[1]]['value'].values
        
        if len(data_a) < 2 or len(data_b) < 2:
            continue
            
        # Mann-Whitney U Test (Non-parametric)
        try:
            _, p_val = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
        except ValueError:
            # Occurs if all numbers are identical
            p_val = 1.0
        
        # Cohen's d (Effect Size)
        d = compute_cohens_d(data_a, data_b)
        
        p_values.append(p_val)
        effect_sizes.append(d)
        valid_layers.append(l)
    
    if not p_values:
        return {}

    # 2. FDR Correction
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    
    # 3. Package Results
    results = {}
    for i, l in enumerate(valid_layers):
        results[l] = {
            'p_raw': p_values[i],
            'p_adj': pvals_corrected[i],
            'significant': reject[i],
            'effect_size': effect_sizes[i]
        }
    return results

def plot_raincloud_robust(df, stats_results, metric_name, output_path):
    """
    Plots distributions with FDR-corrected significance and Effect Size.
    """
    # Filter data for this metric
    subset = df[df['metric'] == metric_name]
    if subset.empty: return

    plt.figure(figsize=(18, 8))
    ax = plt.gca()
    
    # 1. Violin (Density)
    sns.violinplot(data=subset, x='layer', y='value', hue='method', density_norm='width',
                   split=True, inner=None, palette=COLORS, alpha=0.7, ax=ax, linewidth=0, cut=0)
    
    # 2. Box (Stats)
    sns.boxplot(data=subset, x='layer', y='value', hue='method',
                width=0.1, fliersize=0, palette=COLORS, ax=ax, boxprops={'alpha': 0.9})
    
    # 3. Strip (Raw Data) - Optional, commented out to keep it clean as per your preference
    # sns.stripplot(data=subset, x='layer', y='value', hue='method', 
    #               dodge=True, jitter=True, size=1.5, alpha=0.4, palette=COLORS, ax=ax)

    # 4. Stats Annotation
    y_max = subset['value'].max()
    y_min = subset['value'].min()
    y_range = y_max - y_min
    if y_range == 0: y_range = 1.0
    offset = y_range * 0.08
    
    layers = sorted(subset['layer'].unique())
    
    for i, l in enumerate(layers):
        if l not in stats_results: continue
        res = stats_results[l]
        
        is_sig = res['significant']
        eff_size = res['effect_size']
        p_adj = res['p_adj']
        
        if is_sig:
            if p_adj < 0.001: star = "***"
            elif p_adj < 0.01: star = "**"
            else: star = "*"
            
            # Color logic: 
            # Blue (Filter > Pattern), Red (Pattern > Filter)
            if eff_size > 0.8:
                color = '#A3D5FF' 
                fontweight = 'bold'
            elif eff_size < -0.8:
                color = '#FF7886'
                fontweight = 'bold'
            else:
                color = 'black'
                fontweight = 'normal'
            
            label = f"{star}\nd={eff_size:.2f}"
            ax.text(i, y_max + offset, label, ha='center', va='bottom', 
                    color=color, fontsize=9, fontweight=fontweight)
        else:
            ax.text(i, y_max + offset, "ns", ha='center', va='bottom', color='gray', fontsize=8)

    # Formatting
    handles, labels = ax.get_legend_handles_labels()
    # Only show the violin legends
    ax.legend(handles[:2], labels[:2], title="Method", loc='upper left') 
    
    ax.set_xlabel("Transformer Layer", fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Baselines
    if metric_name == "TCAV Score":
        ax.axhline(0.5, linestyle="--", linewidth=1, color="gray", alpha=0.7, zorder=0)
    elif metric_name in ["Cosine Similarity", "Sensitivity"]:
        ax.axhline(0.0, linestyle="--", linewidth=1, color="gray", alpha=0.7, zorder=0)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Adjust Y limit to fit annotations
    ax.set_ylim(top=y_max + y_range * 0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

def main(args):
    print(f"Comparing Final Metrics: Filter vs Pattern | Concept: {args.concept_name}")
    
    # 1. Load Data
    df = load_data(args)
    if df.empty:
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    # List of metrics to process
    target_metrics = ['TCAV Score', 'Cosine Similarity', 'Sensitivity']
    
    summary_records = []
    
    for metric in target_metrics:
        print(f"\n--- Processing {metric} ---")
        
        # Run Stats
        stats_res = run_statistics(df, metric, group_col='method', groups=['Filter', 'Pattern'])
        
        # Generate Plot
        safe_name = metric.lower().replace(" ", "_")
        plot_path = os.path.join(args.output_dir, f"compare_final_{safe_name}.png")
        plot_raincloud_robust(df, stats_res, metric, plot_path)
        
        # Collect Stats for CSV
        layers = sorted(df[df['metric'] == metric]['layer'].unique())
        for l in layers:
            if l in stats_res:
                r = stats_res[l]
                
                # Calculate means for the CSV summary
                l_data = df[(df['layer'] == l) & (df['metric'] == metric)]
                mean_filter = l_data[l_data['method']=='Filter']['value'].mean()
                mean_pattern = l_data[l_data['method']=='Pattern']['value'].mean()
                
                summary_records.append({
                    'Layer': l,
                    'Metric': metric,
                    'Mean_Filter': mean_filter,
                    'Mean_Pattern': mean_pattern,
                    'P_Raw': r['p_raw'],
                    'P_Adj': r['p_adj'],
                    'Significant': r['significant'],
                    'Cohen_d': r['effect_size']
                })

    # Save Stats Summary to CSV
    csv_path = os.path.join(args.output_dir, f"final_stats_summary_{args.concept_name}.csv")
    pd.DataFrame(summary_records).to_csv(csv_path, index=False)
    print(f"\nFull statistics saved to: {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing final_metrics_summary_{concept}.json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--concept_name", type=str, default="concept_set")
    
    args = parser.parse_args()
    main(args)