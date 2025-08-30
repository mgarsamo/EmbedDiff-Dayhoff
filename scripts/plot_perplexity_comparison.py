#!/usr/bin/env python3
"""
Perplexity Comparison Plotting Script

This script creates comprehensive publication-quality plots comparing perplexity scores
between ESM-2 and Dayhoff generated sequences.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
import os


def create_comprehensive_perplexity_plots(csv_path="figures/perplexity_scores.csv", output_dir="figures"):
    """
    Create comprehensive perplexity comparison plots.
    
    Args:
        csv_path (str): Path to the perplexity scores CSV file
        output_dir (str): Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    print(f"📊 Loading perplexity scores from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} perplexity scores")
    print(f"📋 Models: {df['model'].unique()}")
    
    # Set up publication-quality styling
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    
    # Colors for each model
    colors = ['#FF6B6B', '#4ECDC4']  # Red for ESM-2, Teal for Dayhoff
    model_names = ['ESM-2', 'Dayhoff']
    
    # Create comprehensive 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Boxplot (linear scale)
    ax1 = axes[0, 0]
    bp = ax1.boxplot([df[df['model'] == 'ESM-2']['perplexity'], 
                     df[df['model'] == 'Dayhoff']['perplexity']],
                    labels=model_names,
                    patch_artist=True,
                    boxprops=dict(facecolor='white', alpha=0.8, linewidth=1.2),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Perplexity Score', fontsize=14, fontweight='bold')
    ax1.set_title('Perplexity Distribution Comparison (Linear Scale)', fontsize=16, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Boxplot (log scale)
    ax2 = axes[0, 1]
    bp_log = ax2.boxplot([df[df['model'] == 'ESM-2']['perplexity'], 
                         df[df['model'] == 'Dayhoff']['perplexity']],
                        labels=model_names,
                        patch_artist=True,
                        boxprops=dict(facecolor='white', alpha=0.8, linewidth=1.2),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2))
    
    # Color the boxes
    for patch, color in zip(bp_log['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Perplexity Score (log10)', fontsize=14, fontweight='bold')
    ax2.set_title('Perplexity Distribution Comparison (Log Scale)', fontsize=16, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. Histogram with KDE (linear scale)
    ax3 = axes[1, 0]
    for i, model in enumerate(model_names):
        model_data = df[df['model'] == model]['perplexity']
        if len(model_data) > 0:
            # Histogram
            ax3.hist(model_data, bins=30, alpha=0.6, color=colors[i], 
                    label=f'{model} (n={len(model_data)})', density=True)
            
            # KDE curve
            kde = gaussian_kde(model_data)
            x_range = np.linspace(model_data.min(), model_data.max(), 100)
            ax3.plot(x_range, kde(x_range), color=colors[i], linewidth=2, 
                    label=f'{model} KDE')
    
    ax3.set_xlabel('Perplexity Score', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax3.set_title('Perplexity Distribution Histograms + KDE (Linear Scale)', fontsize=16, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 4. Histogram with KDE (log scale)
    ax4 = axes[1, 1]
    for i, model in enumerate(model_names):
        model_data = df[df['model'] == model]['perplexity']
        if len(model_data) > 0:
            # Log-transformed data for histogram
            log_data = np.log10(model_data)
            
            # Histogram
            ax4.hist(log_data, bins=30, alpha=0.6, color=colors[i], 
                    label=f'{model} (n={len(model_data)})', density=True)
            
            # KDE curve on log scale
            kde = gaussian_kde(log_data)
            x_range = np.linspace(log_data.min(), log_data.max(), 100)
            ax4.plot(x_range, kde(x_range), color=colors[i], linewidth=2, 
                    label=f'{model} KDE')
    
    ax4.set_xlabel('log10(Perplexity Score)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax4.set_title('Perplexity Distribution Histograms + KDE (Log Scale)', fontsize=16, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Adjust layout and save
    plt.tight_layout()
    comprehensive_plot_path = os.path.join(output_dir, "perplexity_comprehensive_comparison.png")
    plt.savefig(comprehensive_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Comprehensive perplexity plots saved to: {comprehensive_plot_path}")
    plt.close()
    
    # Create individual plots for flexibility
    create_individual_perplexity_plots(df, output_dir, colors, model_names)
    
    # Print summary statistics
    print_summary_statistics(df, model_names)
    
    return comprehensive_plot_path


def create_individual_perplexity_plots(df, output_dir, colors, model_names):
    """Create individual perplexity plots for flexibility."""
    
    # 1. Simple boxplot
    plt.figure(figsize=(8, 6))
    bp = plt.boxplot([df[df['model'] == 'ESM-2']['perplexity'], 
                     df[df['model'] == 'Dayhoff']['perplexity']],
                    labels=model_names,
                    patch_artist=True,
                    boxprops=dict(facecolor='white', alpha=0.8, linewidth=1.2),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('Perplexity Score', fontsize=14, fontweight='bold')
    plt.title('Perplexity Distribution Comparison (Dayhoff)', fontsize=16, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    boxplot_path = os.path.join(output_dir, "perplexity_boxplot_dayhoff.png")
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📦 Boxplot saved to: {boxplot_path}")
    
    # 2. Log-scale boxplot
    plt.figure(figsize=(8, 6))
    bp_log = plt.boxplot([df[df['model'] == 'ESM-2']['perplexity'], 
                         df[df['model'] == 'Dayhoff']['perplexity']],
                        labels=model_names,
                        patch_artist=True,
                        boxprops=dict(facecolor='white', alpha=0.8, linewidth=1.2),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2))
    
    for patch, color in zip(bp_log['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('Perplexity Score (log10)', fontsize=14, fontweight='bold')
    plt.title('Perplexity Distribution Comparison - Log Scale (Dayhoff)', fontsize=16, fontweight='bold')
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    log_boxplot_path = os.path.join(output_dir, "perplexity_log_boxplot_dayhoff.png")
    plt.savefig(log_boxplot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📊 Log-scale boxplot saved to: {log_boxplot_path}")


def print_summary_statistics(df, model_names):
    """Print comprehensive summary statistics."""
    print("\n" + "="*60)
    print("PERPLEXITY SCORING SUMMARY STATISTICS")
    print("="*60)
    
    for model in model_names:
        model_data = df[df['model'] == model]['perplexity']
        if len(model_data) > 0:
            mean_perp = model_data.mean()
            std_perp = model_data.std()
            median_perp = model_data.median()
            min_perp = model_data.min()
            max_perp = model_data.max()
            print(f"{model:>8}: {mean_perp:.4f} ± {std_perp:.4f} (n={len(model_data)})")
            print(f"{'':>8}  Median: {median_perp:.4f}, Range: [{min_perp:.4f}, {max_perp:.4f}]")
        else:
            print(f"{model:>8}: No data available")
    
    print("="*60)


def main():
    """Main function to run the perplexity plotting analysis."""
    # Check if perplexity scores exist
    csv_path = "figures/perplexity_scores.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Perplexity scores file not found: {csv_path}")
        print("Please run the perplexity scoring script first:")
        print("python scripts/perplexity_scoring.py")
        return
    
    try:
        # Create comprehensive plots
        print("🎨 Creating comprehensive perplexity comparison plots...")
        plot_path = create_comprehensive_perplexity_plots(csv_path)
        
        print(f"\n🎉 Perplexity plotting completed successfully!")
        print(f"📊 All plots saved to: figures/")
        
    except Exception as e:
        print(f"❌ Error during plotting: {e}")
        raise


if __name__ == "__main__":
    main()
