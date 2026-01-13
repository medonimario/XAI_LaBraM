import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import argparse
from tqdm import tqdm

# --- CONFIGURATION ---
# Matching the pastel/clean aesthetic
COLORS = {
    'Filter': '#A3D5FF',  # Bright Blue
    'Pattern': '#FF7886'  # Soft Red
}

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
            print(f"Metrics file not found: {json_path}")
            return pd.DataFrame()
            
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
            
            # Normalize vectors to be safe
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms==0] = 1e-9
            vecs_norm = vecs / norms
            
            # Compute Mean Direction
            mean_vec = np.mean(vecs_norm, axis=0)
            mean_vec /= (np.linalg.norm(mean_vec) + 1e-9)
            
            # Compute Cosines
            cosines = vecs_norm @ mean_vec
            
            # B. Aggregate Metrics
            metrics_list = layer_m[method]
            
            for i in range(len(metrics_list)):
                rec = {
                    'layer': layer_id,
                    'method': method_key,
                    'run': i,
                    'consistency': float(cosines[i]),
                    'test_acc': metrics_list[i]['test_acc'],
                    'test_auc': metrics_list[i]['test_auc']
                }
                data_records.append(rec)

    return pd.DataFrame(data_records)

# --- Plotting Helpers ---

def add_significance_stars(ax, df, x_col, y_col, hue_col, pairs):
    """
    Calculates Mann-Whitney U test and adds bars/stars to the plot.
    """
    y_max = df[y_col].max()
    y_range = y_max - df[y_col].min()
    if y_range == 0: y_range = 1.0
    offset = y_range * 0.06
    
    # Iterate over x-axis categories (Layers)
    layers = sorted(df[x_col].unique())
    
    for i, layer in enumerate(layers):
        layer_data = df[df[x_col] == layer]
        
        # Get data for the two groups
        group_a = layer_data[layer_data[hue_col] == pairs[0]][y_col]
        group_b = layer_data[layer_data[hue_col] == pairs[1]][y_col]
        
        if len(group_a) == 0 or len(group_b) == 0: continue

        # Statistical Test
        try:
            stat, p_val = mannwhitneyu(group_a, group_b, alternative='two-sided')
        except ValueError:
            p_val = 1.0
        
        # Determine Stars
        if p_val < 0.001: star = "***"
        elif p_val < 0.01: star = "**"
        elif p_val < 0.05: star = "*"
        else: star = "ns"
        
        # Calculate max y at this x position for placement
        curr_y = layer_data[y_col].max()
        
        if star != "ns":
            color = 'black'
            weight = 'bold'
        else:
            color = 'gray'
            weight = 'normal'

        ax.text(i, curr_y + offset, star, ha='center', va='bottom', 
                color=color, fontsize=8, fontweight=weight)

def plot_raincloud(df, x_col, y_col, hue_col, title, output_path, y_label):
    """
    Creates a Violin + Strip + Box plot with significance stars.
    Visual style matched to the 'robust' Null comparison script.
    """
    plt.figure(figsize=(18, 8))
    ax = plt.gca()
    
    # 1. Violin (Density) - Matched Style
    sns.violinplot(data=df, x=x_col, y=y_col, hue=hue_col, density_norm='width',
                   split=True, inner=None, palette=COLORS, alpha=0.7, ax=ax, linewidth=0, cut=0)
    
    # 2. Box (Stats) - Matched Style
    sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col,
                width=0.1, fliersize=0, palette=COLORS, ax=ax, boxprops={'alpha': 0.9})
    
    # 3. Strip (Raw Data) - Jittered
    sns.stripplot(data=df, x=x_col, y=y_col, hue=hue_col, 
                  dodge=True, jitter=True, size=2, alpha=0.9, palette=COLORS, ax=ax)

    # 4. Significance
    add_significance_stars(ax, df, x_col, y_col, hue_col, ['Filter', 'Pattern'])

    # Formatting
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="Method", loc='upper left') 
    
    # Clean Aesthetics
    # ax.set_title(title, fontsize=16) # Optional: Remove title for cleaner paper figures
    ax.set_xlabel("Transformer Layer", fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Remove Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Baselines for Accuracy/AUC
    if "Accuracy" in y_label or "AUC" in y_label:
        ax.axhline(0.5, linestyle="--", linewidth=1, color="gray", alpha=0.7, zorder=0)

    # Adjust Y limit to fit stars
    y_max = df[y_col].max()
    y_range = y_max - df[y_col].min()
    if y_range == 0: y_range = 1.0
    ax.set_ylim(top=y_max + y_range * 0.15)
    
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
    cols = 4
    rows = int(np.ceil(n_layers / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
    axes = axes.flatten()
    
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
        
        # Zoom logic
        ax.set_xlim(0.8, 1.0) 
        
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

def main(args):
    # 1. Load Data
    df = load_data(args)
    if df.empty:
        print("No data found. Check directories.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Plot Consistency (Histograms)
    plot_consistency_histograms(
        df, 
        os.path.join(args.output_dir, "comparison_1_consistency.png")
    )

    # 3. Plot Accuracy (Raincloud)
    plot_raincloud(
        df, x_col='layer', y_col='test_acc', hue_col='method',
        title=f"Generalization Accuracy: Filter vs Pattern (Concept: {args.concept_name})",
        output_path=os.path.join(args.output_dir, "comparison_2_accuracy.png"),
        y_label="Test Accuracy"
    )

    # 4. Plot AUC (Raincloud)
    plot_raincloud(
        df, x_col='layer', y_col='test_auc', hue_col='method',
        title=f"Generalization AUC: Filter vs Pattern (Concept: {args.concept_name})",
        output_path=os.path.join(args.output_dir, "comparison_3_auc.png"),
        y_label="Test AUC"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing cavs_*.pkl and metrics_*.json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--concept_name", type=str, default="concept_set")
    parser.add_argument("--target_layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    
    args = parser.parse_args()
    main(args)