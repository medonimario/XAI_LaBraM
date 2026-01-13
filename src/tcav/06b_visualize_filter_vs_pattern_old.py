import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import argparse
from tqdm import tqdm

# --- CONFIGURATION ---
COLORS = {
    'Filter': '#A3D5FF',  # Bright Blue
    'Pattern': '#FF7886'  # Soft Red
}

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
    layer_ids = sorted([int(k) for k in data.keys()])
    
    for layer_id in layer_ids:
        layer_data = data[str(layer_id)]
        
        for method in ['filter', 'pattern']:
            method_label = 'Filter' if method == 'filter' else 'Pattern'
            metrics = layer_data[method]
            
            scores = metrics['tcav_scores']
            sensitivities = metrics['mean_sensitivities']
            cosines = metrics['mean_cosines']
            
            for i in range(len(scores)):
                records.append({'layer': layer_id, 'method': method_label, 'metric': 'TCAV Score', 'value': scores[i]})
                records.append({'layer': layer_id, 'method': method_label, 'metric': 'Sensitivity', 'value': sensitivities[i]})
                records.append({'layer': layer_id, 'method': method_label, 'metric': 'Cosine Similarity', 'value': cosines[i]})

    return pd.DataFrame(records)

# --- Plotting Helpers ---

def add_significance_stars(ax, df, x_col, y_col, hue_col, pairs):
    """
    Calculates Mann-Whitney U test and adds significance annotations.
    """
    y_max = df[y_col].max()
    y_range = y_max - df[y_col].min()
    if y_range == 0: y_range = 1.0
        
    offset = y_range * 0.06
    
    layers = sorted(df[x_col].unique())
    
    for i, layer in enumerate(layers):
        layer_data = df[df[x_col] == layer]
        
        group_a = layer_data[layer_data[hue_col] == pairs[0]][y_col]
        group_b = layer_data[layer_data[hue_col] == pairs[1]][y_col]
        
        if len(group_a) < 2 or len(group_b) < 2: 
            continue

        try:
            stat, p_val = mannwhitneyu(group_a, group_b, alternative='two-sided')
        except ValueError:
            p_val = 1.0
        
        if p_val < 0.001: star = "***"
        elif p_val < 0.01: star = "**"
        elif p_val < 0.05: star = "*"
        else: star = "ns"
        
        curr_max = layer_data[y_col].max()
        
        if star != "ns":
            color = 'black'
            weight = 'bold'
        else:
            color = 'gray'
            weight = 'normal'
            
        ax.text(i, curr_max + offset, star, ha='center', va='bottom', 
                color=color, fontsize=8, fontweight=weight)

def plot_raincloud(df, metric_name, output_path):
    """
    Generates a Raincloud Plot (Violin + Box + Strip) for a specific metric.
    Styled to match the Robust Real vs Null plots.
    """
    subset = df[df['metric'] == metric_name]
    
    if subset.empty:
        print(f"No data for metric: {metric_name}")
        return

    plt.figure(figsize=(18, 8))
    ax = plt.gca()
    
    # 1. Violin Plot (Distribution Shape) - Matched Style
    sns.violinplot(data=subset, x='layer', y='value', hue='method', density_norm='width',
                   split=True, inner=None, palette=COLORS, alpha=0.7, ax=ax, linewidth=0, cut=0)
    
    # 2. Box Plot (Quartiles) - Matched Style
    sns.boxplot(data=subset, x='layer', y='value', hue='method',
                width=0.1, fliersize=0, palette=COLORS, ax=ax, boxprops={'alpha': 0.9})
    
    # 3. Strip Plot (Optional: Removed to reduce clutter as per previous request, or keep minimal)
    # If you want points back, uncomment below:
    # sns.stripplot(data=subset, x='layer', y='value', hue='method', 
    #               dodge=True, jitter=True, size=2, alpha=0.5, palette=COLORS, ax=ax)

    # 4. Significance Testing
    add_significance_stars(ax, subset, 'layer', 'value', 'method', ['Filter', 'Pattern'])

    # Formatting
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="Method", loc='upper left')
    
    # ax.set_title(f"Comparison: {metric_name} (Filter vs. Pattern)", fontsize=16)
    ax.set_xlabel("Transformer Layer", fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Clean Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Baselines
    if metric_name == "TCAV Score":
        ax.axhline(0.5, linestyle="--", linewidth=1, color="gray", alpha=0.7, zorder=0)
    elif metric_name in ["Cosine Similarity", "Sensitivity"]:
        ax.axhline(0.0, linestyle="--", linewidth=1, color="gray", alpha=0.7, zorder=0)

    # Adjust Y-limits slightly to fit stars
    y_max = subset['value'].max()
    y_min = subset['value'].min()
    y_range = y_max - y_min
    if y_range == 0: y_range = 1.0
    
    ax.set_ylim(top=y_max + y_range * 0.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

def main(args):
    print(f"Loading final metrics for concept: {args.concept_name}")
    df = load_data(args)
    
    if df.empty:
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Plot 1: TCAV Scores
    plot_raincloud(df, 'TCAV Score', os.path.join(args.output_dir, "compare_final_1_tcav_score.png"))
    
    # 3. Plot 2: Cosine Similarity
    plot_raincloud(df, 'Cosine Similarity', os.path.join(args.output_dir, "compare_final_2_cosine.png"))
    
    # 4. Plot 3: Sensitivity
    plot_raincloud(df, 'Sensitivity', os.path.join(args.output_dir, "compare_final_3_sensitivity.png"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Filter vs Pattern Final Metrics")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing final_metrics_summary_{concept}.json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--concept_name", type=str, default="concept_set")
    
    args = parser.parse_args()
    main(args)