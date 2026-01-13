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
    Crawls the output directory to gather Vectors (for consistency) 
    and Metrics (Accuracy/AUC) into a single DataFrame.
    """
    data_records = []
    target_layers = [int(l) for l in args.target_layers.split(',')]

    print("Loading data...")
    for layer_id in tqdm(target_layers):
        # 1. Load Metrics JSON
        json_path = os.path.join(args.input_dir, f"metrics_{args.concept_name}.json")
        if not os.path.exists(json_path):
            # Try skipping if specific file missing, or return empty if essential
            continue
            
        with open(json_path, 'r') as f:
            all_metrics = json.load(f)
            
        if str(layer_id) not in all_metrics:
            continue
            
        layer_m = all_metrics[str(layer_id)]

        # 2. Load Vectors (for Consistency Calculation)
        vec_path = os.path.join(args.input_dir, f"cavs_{args.concept_name}_layer_{layer_id}.pkl")
        if not os.path.exists(vec_path):
            continue
            
        with open(vec_path, 'rb') as f:
            vectors_dict = pickle.load(f)

        # --- Process Data for this Layer ---
        for method in ['filter', 'pattern']:
            method_key = 'Filter' if method == 'filter' else 'Pattern'
            
            # A. Calculate Consistency (Cosine to Mean)
            vecs = np.array(vectors_dict[method]) # Shape: [N_runs, Dim]
            
            # Normalize vectors
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms==0] = 1e-9
            vecs_norm = vecs / norms
            
            # Compute Mean Direction
            mean_vec = np.mean(vecs_norm, axis=0)
            mean_vec /= (np.linalg.norm(mean_vec) + 1e-9)
            
            # Compute Cosines
            cosines = vecs_norm @ mean_vec
            
            # B. Aggregate Metrics
            # Check if key exists
            if method not in layer_m: continue
            metrics_list = layer_m[method]
            
            # Limit to min length between vectors and metrics to avoid index errors
            n_samples = min(len(cosines), len(metrics_list))
            
            for i in range(n_samples):
                rec = {
                    'layer': layer_id,
                    'method': method_key,
                    'run': i,
                    'consistency': float(cosines[i]),
                    'test_acc': metrics_list[i].get('test_acc', 0),
                    'test_auc': metrics_list[i].get('test_auc', 0.5)
                }
                data_records.append(rec)

    return pd.DataFrame(data_records)

def run_statistics(df, metric_col, group_col='method', groups=['Filter', 'Pattern'], alpha=0.05):
    """
    Runs Mann-Whitney U test per layer, calculates Cohen's d, 
    and applies FDR (Benjamini-Hochberg) correction.
    """
    layers = sorted(df['layer'].unique())
    p_values = []
    effect_sizes = []
    valid_layers = []

    # 1. Compute Raw Stats
    for l in layers:
        layer_data = df[df['layer'] == l]
        
        # Extract the two distributions
        # Note: We calculate (Group 0 - Group 1). 
        # So if groups=['Filter', 'Pattern'], d > 0 means Filter > Pattern.
        data_a = layer_data[layer_data[group_col] == groups[0]][metric_col].values
        data_b = layer_data[layer_data[group_col] == groups[1]][metric_col].values
        
        if len(data_a) < 2 or len(data_b) < 2:
            continue
            
        # Mann-Whitney U Test (Non-parametric)
        _, p_val = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
        
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

def plot_raincloud_robust(df, stats_results, x_col, y_col, hue_col, title, output_path, y_label):
    """
    Plots distributions with FDR-corrected significance and Effect Size.
    """
    if df.empty: return

    plt.figure(figsize=(18, 8))
    ax = plt.gca()
    
    # 1. Violin (Density)
    sns.violinplot(data=df, x=x_col, y=y_col, hue=hue_col, density_norm='width',
                   split=True, inner=None, palette=COLORS, alpha=0.7, ax=ax, linewidth=0, cut=0)
    
    # 2. Box (Stats)
    sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col,
                width=0.1, fliersize=0, palette=COLORS, ax=ax, boxprops={'alpha': 0.9})
    
    # 3. Strip (Raw Data)
    # Using smaller size and less alpha to not clutter the effect size text
    # sns.stripplot(data=df, x=x_col, y=y_col, hue=hue_col, 
    #               dodge=True, jitter=True, size=1.5, alpha=0.4, palette=COLORS, ax=ax)

    # 4. Stats Annotation
    y_max = df[y_col].max()
    y_range = y_max - df[y_col].min()
    if y_range == 0: y_range = 1.0
    offset = y_range * 0.08
    
    layers = sorted(df[x_col].unique())
    
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
            # If eff_size is very positive (Filter >> Pattern) or negative (Pattern >> Filter)
            if eff_size > 0.8:
                color = '#A3D5FF' # Blue (Filter color) indicating Filter is higher
                fontweight = 'bold'
            elif eff_size < -0.8:
                color = '#FF7886' # Red (Pattern color) indicating Pattern is higher
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
    # Fix Legend
    handles, labels = ax.get_legend_handles_labels()
    # We only want the first 2 handles (Violins) not the boxplot/stripplot duplicates
    ax.legend(handles[:2], labels[:2], title="Method", loc='upper left') 
    
    # ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Transformer Layer", fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Baselines
    if "Accuracy" in y_label or "AUC" in y_label:
        ax.axhline(0.5, linestyle="--", linewidth=1, color="gray", alpha=0.7, zorder=0)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Adjust Y limit to fit annotations
    ax.set_ylim(top=y_max + y_range * 0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

def plot_consistency_histograms(df, output_path):
    """
    Grid of histograms comparing consistency distributions per layer.
    """
    layers = sorted(df['layer'].unique())
    n_layers = len(layers)
    if n_layers == 0: return

    cols = 4
    rows = int(np.ceil(n_layers / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
    if rows * cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    print("Generating Consistency Histograms...")
    
    for i, layer in enumerate(layers):
        ax = axes[i]
        subset = df[df['layer'] == layer]
        
        # Histograms
        sns.histplot(data=subset, x='consistency', hue='method', 
                     kde=True, element="step", palette=COLORS, ax=ax, bins=20, alpha=0.5)
        
        # Calculate stats for title
        f_mean = subset[subset['method']=='Filter']['consistency'].mean()
        p_mean = subset[subset['method']=='Pattern']['consistency'].mean()
        
        ax.set_title(f"Layer {layer}\nMean: F={f_mean:.3f} | P={p_mean:.3f}", fontsize=10)
        ax.set_xlabel("Cosine Similarity", fontsize=9)
        
        # Clean look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # Zoom logic (Consistency is usually high 0.8-1.0)
        ax.set_xlim(0.8, 1.0) 
        
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

def main(args):
    print(f"Comparing Methods: Filter vs Pattern | Concept: {args.concept_name}")
    
    # 1. Load Data
    df = load_data(args)
    if df.empty:
        print("No data found. Check directories.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Plot Consistency (Histograms) - No statistical stars here usually
    plot_consistency_histograms(
        df, 
        os.path.join(args.output_dir, "comparison_1_consistency.png")
    )
    # 2.5 Analyze and Plot Consistency
    print("\n--- Processing Consistency ---")
    stats_consistency = run_statistics(df, 'consistency', group_col='method', groups=['Filter', 'Pattern'])
    plot_raincloud_robust(
        df, stats_consistency, x_col='layer', y_col='consistency', hue_col='method',
        title=f"Consistency: Filter vs Pattern",
        output_path=os.path.join(args.output_dir, "comparison_1b_consistency_raincloud.png"),
        y_label="Cosine Similarity"
    )

    # 3. Analyze and Plot Accuracy
    print("\n--- Processing Accuracy ---")
    stats_acc = run_statistics(df, 'test_acc', group_col='method', groups=['Filter', 'Pattern'])
    
    plot_raincloud_robust(
        df, stats_acc, x_col='layer', y_col='test_acc', hue_col='method',
        title=f"Generalization Accuracy: Filter vs Pattern",
        output_path=os.path.join(args.output_dir, "comparison_2_accuracy.png"),
        y_label="Test Accuracy"
    )

    # 4. Analyze and Plot AUC
    print("\n--- Processing AUC ---")
    stats_auc = run_statistics(df, 'test_auc', group_col='method', groups=['Filter', 'Pattern'])
    
    plot_raincloud_robust(
        df, stats_auc, x_col='layer', y_col='test_auc', hue_col='method',
        title=f"Generalization AUC: Filter vs Pattern",
        output_path=os.path.join(args.output_dir, "comparison_3_auc.png"),
        y_label="Test AUC"
    )

    # 5. Save Stats Summary to CSV
    summary_records = []
    layers = sorted(df['layer'].unique())
    
    for metric_name, stats_dict in [('Accuracy', stats_acc), ('AUC', stats_auc)]:
        for l in layers:
            if l in stats_dict:
                r = stats_dict[l]
                
                # Get means for context
                l_data = df[df['layer'] == l]
                mean_filter = l_data[l_data['method']=='Filter'][ 'test_acc' if metric_name=='Accuracy' else 'test_auc'].mean()
                mean_pattern = l_data[l_data['method']=='Pattern']['test_acc' if metric_name=='Accuracy' else 'test_auc'].mean()
                
                summary_records.append({
                    'Layer': l,
                    'Metric': metric_name,
                    'Mean_Filter': mean_filter,
                    'Mean_Pattern': mean_pattern,
                    'P_Raw': r['p_raw'],
                    'P_Adj': r['p_adj'],
                    'Significant': r['significant'],
                    'Cohen_d': r['effect_size']
                })
    
    csv_path = os.path.join(args.output_dir, f"stats_summary_filter_vs_pattern_{args.concept_name}.csv")
    pd.DataFrame(summary_records).to_csv(csv_path, index=False)
    print(f"\nStats summary saved to: {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing cavs_*.pkl and metrics_*.json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--concept_name", type=str, default="concept_set")
    parser.add_argument("--target_layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    
    args = parser.parse_args()
    main(args)