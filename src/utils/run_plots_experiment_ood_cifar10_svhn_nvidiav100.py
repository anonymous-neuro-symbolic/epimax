import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def render_ood_sweep_article_format(csv_filepath="dual_ood_sweep_final.csv"):
    # 1. Load the empirical data from the V100 experiment
    if not os.path.exists(csv_filepath):
        print(f"Error: {csv_filepath} not found in the current directory.")
        return
        
    df = pd.read_csv(csv_filepath)

    # 2. Setup the academic plotting style
    # We use a whitegrid for readability, but manually override sizes for publication
    sns.set_theme(style="whitegrid", context="paper")
    
    # Large figure size to give the subplots room to breathe
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    metrics = ['AUROC', 'AUPR', 'FPR95']
    titles = ['AUROC (Higher is Better)', 'AUPR (Higher is Better)', 'FPR95 (Lower is Better)']

    # 3. Plot each metric
    for i, metric in enumerate(metrics):
        # Lineplot automatically calculates the mean and standard deviation (sd) across the 10 rounds
        sns.lineplot(
            data=df, 
            x='gamma', 
            y=metric, 
            hue='ood_dataset', 
            marker='o', 
            markersize=12,      # Large markers
            linewidth=3,        # Thick lines
            errorbar='sd',      # Standard deviation bands
            err_style='band', 
            ax=axes[i]
        )
        
        # ARTICLE FORMATTING: Massive font sizes and padding
        axes[i].set_title(titles[i], fontsize=26, pad=20, fontweight='bold')
        axes[i].set_xlabel(r'Focal Pressure ($\gamma$)', fontsize=22, labelpad=15)
        axes[i].set_ylabel(metric, fontsize=22, labelpad=15)
        
        # Increase tick label sizes
        axes[i].tick_params(axis='both', which='major', labelsize=20)
        
        # Handle the Legend (Only need it on the first plot to save space)
        if i == 0:
            axes[i].legend(
                title='OOD Dataset', 
                title_fontsize=22, 
                fontsize=20, 
                loc='lower right' if metric != 'FPR95' else 'upper right'
            )
        else:
            axes[i].get_legend().remove()

    # 4. Master Title and Layout
    plt.suptitle(
        r'EpiMax Out-of-Distribution Separability vs. Focal Pressure ($\gamma$)', 
        fontsize=30, 
        fontweight='bold', 
        y=1.08
    )
    
    plt.tight_layout()
    
    # 5. Save the high-resolution image
    output_filename = 'ood_sweep_article_format.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Article-formatted OOD sweep chart successfully saved as: {output_filename}")

if __name__ == "__main__":
    render_ood_sweep_article_format()