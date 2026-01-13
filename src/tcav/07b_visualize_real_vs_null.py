import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import argparse
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# --- CONFIGURATION ---
COLORS = {
    # 'Real': '#A3D5FF',   # Bright Blue
    'Real': '#FF7886',    # Soft Red
    'Null': '#B8B8B8'    # Gray (Background)
}

def compute_cohens_d(x, y):
    """Computes Cohen's d (Effect Size) for two independent samples."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pool_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
    if pool_std == 0: return 0
    return (np.mean(x) - np.mean(y)) / pool_std

def load_data_for_layer(args, layer_id):
    """
    Loads Real (100 runs) and Pooled Null (~9900 runs) distributions.
    """
    cav_type = args.cav_type  # 'filter' or 'pattern'

    # 1. Load Real Data
    real_pkl = os.path.join(args.real_dir, f"final_metrics_distributions_{args.concept_name}.pkl")
    if not os.path.exists(real_pkl): return None
    
    with open(real_pkl, 'rb') as f:
        data = pickle.load(f)
        if layer_id not in data: return None
        # Dynamically select 'filter' or 'pattern'
        real_full = data[layer_id][cav_type]
        
    # Flatten/Process Real Data
    # Real data contains full arrays of sensitivities/cosines per run
    
    # 1. TCAV Score (Calculated from sensitivities)
    real_tcav = []
    for x in real_full['sensitivities']:
        # If sensitivity vector has magnitude
        if np.sum(np.abs(x) > 1e-9) > 0:
            real_tcav.append(float(np.mean(x > 0)))
        else:
            real_tcav.append(0.5)
            
    # 2. Sensitivity (Mean of the array)
    real_sens = [float(np.mean(x)) for x in real_full['sensitivities']]
    
    # 3. Cosine (Mean of the array)
    real_cos  = [float(np.mean(x)) for x in real_full['cosines']]

    # 2. Load Null Data
    null_pkl = os.path.join(args.null_dir, f"null_metrics_layer_{layer_id}.pkl")
    if not os.path.exists(null_pkl): return None

    with open(null_pkl, 'rb') as f:
        null_data = pickle.load(f)

    # Flatten Null Data (Pooled Universe)
    null_tcav = []
    null_sens = []
    null_cos = []
    
    for anchor_id, data in null_data.items():
        # Dynamically select 'filter' or 'pattern'
        metrics = data[cav_type]
        
        null_tcav.extend(metrics['tcav_scores'])
        null_sens.extend(metrics['mean_sens'])
        null_cos.extend(metrics['mean_cos'])

    return {
        'TCAV Score': (real_tcav, null_tcav),
        'Sensitivity': (real_sens, null_sens),
        'Cosine Similarity': (real_cos, null_cos)
    }

def run_statistics(data_registry, metric_name, alpha=0.05):
    """Runs Welch's T-test and FDR correction."""
    layers = sorted(data_registry.keys())
    p_values = []
    effect_sizes = []
    
    for l in layers:
        real_dist, null_dist = data_registry[l][metric_name]
        
        # Welch's T-Test
        _, p_val = stats.ttest_ind(real_dist, null_dist, equal_var=False, alternative='two-sided')
        d = compute_cohens_d(real_dist, null_dist)
        
        p_values.append(p_val)
        effect_sizes.append(d)
        
    # FDR Correction
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    
    results = {}
    for i, l in enumerate(layers):
        results[l] = {
            'p_adj': pvals_corrected[i],
            'significant': reject[i],
            'effect_size': effect_sizes[i]
        }
    return results

def plot_raincloud_robust(data_registry, stats_results, metric_name, cav_type, output_path):
    """
    Plots distributions with FDR-corrected significance and Effect Size.
    """
    records = []
    layers = sorted(data_registry.keys())
    
    for l in layers:
        real, null = data_registry[l][metric_name]
        # Subsample Nulls for visual clarity (max 2000)
        null_plot = np.random.choice(null, min(len(null), 2000), replace=False)
        
        for v in real: records.append({'layer': l, 'group': 'Real', 'value': v})
        for v in null_plot: records.append({'layer': l, 'group': 'Null', 'value': v})
            
    df = pd.DataFrame(records)
    if df.empty: return

    plt.figure(figsize=(18, 8))
    ax = plt.gca()
    
    # 1. Violin
    sns.violinplot(data=df, x='layer', y='value', hue='group', density_norm='width',
                   split=True, inner=None, palette=COLORS, alpha=0.7, ax=ax, linewidth=0, cut=0)
    
    # 2. Box
    sns.boxplot(data=df, x='layer', y='value', hue='group',
                width=0.1, fliersize=0, palette=COLORS, ax=ax, boxprops={'alpha': 0.9})

    # 3. Stats Annotation
    y_max = df['value'].max()
    y_range = y_max - df['value'].min()
    if y_range == 0: y_range = 1.0
    offset = y_range * 0.06
    
    for i, l in enumerate(layers):
        if l not in stats_results: continue
        res = stats_results[l]
        
        is_sig = res['significant']
        eff_size = res['effect_size']
        
        if is_sig:
            if res['p_adj'] < 0.001: star = "***"
            elif res['p_adj'] < 0.01: star = "**"
            else: star = "*"
            
            # Green for Good (Higher Acc/AUC/Consistency), Red for Bad
            if eff_size > 0.8:
                color = '#8CB369' # Green
                fontweight = 'bold'
            elif eff_size < -0.8:
                color = '#D72638' # Red
                fontweight = 'bold'
            else:
                color = 'black'
                fontweight = 'normal'
            
            label = f"{star}\nd={eff_size:.2f}"
            ax.text(i, y_max + offset, label, ha='center', va='bottom', 
                    color=color, fontsize=8, fontweight=fontweight)
        else:
            ax.text(i, y_max + offset, "ns", ha='center', va='bottom', color='gray', fontsize=8)

    # Formatting
    ax.legend(handles=ax.get_legend_handles_labels()[0][:2], labels=['Real', 'Null'], loc='upper left')
    # Use title to indicate which CAV type was used
    # ax.set_title(f"Real vs. Null ({cav_type.title()} CAV): {metric_name}", fontsize=14, pad=20)
    ax.set_xlabel("Transformer Layer", fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Baselines
    if metric_name == "TCAV Score":
        ax.axhline(0.5, linestyle="--", linewidth=1, color="gray", alpha=0.7, zorder=0)
    else:
        ax.axhline(0.0, linestyle="--", linewidth=1, color="gray", alpha=0.7, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if metric_name == "TCAV Score":
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

def main(args):
    print(f"Comparing Distributions for Concept: {args.concept_name}")
    print(f"Mode: {args.cav_type.upper()} CAV")
    
    target_layers = [int(l) for l in args.target_layers.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    data_registry = {}
    print("Loading data...")
    for layer_id in tqdm(target_layers):
        data = load_data_for_layer(args, layer_id)
        if data:
            data_registry[layer_id] = data
            
    if not data_registry:
        print("No data loaded. Check input directories.")
        return

    # 2. Process Metrics
    metrics = ['TCAV Score', 'Cosine Similarity', 'Sensitivity']
    
    # --- NEW: Initialize list to store CSV rows ---
    stats_summary = []

    print(f"\n{'Layer':<5} | {'Metric':<15} | {'P-Adj':<10} | {'Effect (d)':<10} | {'Result'}")
    print("-" * 60)
    
    for m in metrics:
        stats_results = run_statistics(data_registry, m, alpha=0.05)
        
        # Log and Collect Data
        for l in target_layers:
            if l in stats_results:
                r = stats_results[l]
                
                # --- NEW: Calculate Means for the CSV ---
                real_dist, null_dist = data_registry[l][m]
                real_mean = np.mean(real_dist)
                null_mean = np.mean(null_dist)

                # Determine Result String
                if r['significant'] and r['effect_size'] > 0.8: res_str = "VALID (+)"
                elif r['significant'] and r['effect_size'] < -0.8: res_str = "ANTI (-)"
                elif r['significant']: res_str = "SIG (Small)"
                else: res_str = "NS"
                
                print(f"{l:<5} | {m:<15} | {r['p_adj']:.1e} | {r['effect_size']:<10.2f} | {res_str}")
                
                # --- NEW: Append to summary list ---
                stats_summary.append({
                    'Layer': l,
                    'Metric': m,
                    'Real_Mean': real_mean,
                    'Null_Mean': null_mean,
                    'P_Adjusted': r['p_adj'],
                    'Effect_Size_Cohen_d': r['effect_size'],
                    'Is_Significant': r['significant'],
                    'Result_Category': res_str
                })
        
        # Plot (filename now includes cav_type)
        safe_metric_name = m.replace(" ", "_").lower()
        output_filename = f"{args.cav_type}_{safe_metric_name}.png"
        
        plot_raincloud_robust(
            data_registry, 
            stats_results, 
            m, 
            args.cav_type,
            os.path.join(args.output_dir, output_filename)
        )

    # --- NEW: Save to CSV ---
    csv_filename = f"stats_summary_{args.concept_name}_{args.cav_type}.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)
    pd.DataFrame(stats_summary).to_csv(csv_path, index=False)
    print(f"\nFull statistics saved to: {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, required=True, 
                        help="Dir containing metrics_{concept}.json and cavs_{concept}_*.pkl")
    parser.add_argument("--null_dir", type=str, required=True, 
                        help="Dir containing null_distribution_layer_X.pkl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--concept_name", type=str, default="concept_set")
    parser.add_argument("--target_layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    
    # New Argument
    parser.add_argument("--cav_type", type=str, default="filter", choices=['filter', 'pattern'],
                        help="Type of CAV to analyze: 'filter' or 'pattern'")
    
    args = parser.parse_args()
    main(args)