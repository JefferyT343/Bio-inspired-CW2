"""
Plot multiple TwoStageEvolution simulation results.
Plots all specified CSV files and saves them to replot_results directory.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create output directory
output_dir = "replot_results"
os.makedirs(output_dir, exist_ok=True)

# Define CSV files to plot
csv_files = [
    "TwoStageEvolutionWithShortSR-Distance_200gens.csv",
    "SensorAddtoPredTwoStageEvolution_200gens.csv",
    "ResourceDepletion_200gens.csv",
    "400_ResourceDepletion_200gens.csv"
]

results_dir = "results"

def plot_results(csv_file):
    """Generate plot for a single CSV file."""
    csv_path = os.path.join(results_dir, csv_file)

    if not os.path.exists(csv_path):
        print(f"Warning: Could not find {csv_path}, skipping...")
        return

    print(f"Processing {csv_file}...")

    # Read data
    df = pd.read_csv(csv_path)

    # Identify stage transition
    stage1_end = df[df['Stage'] == 1]['Generation'].max()

    # Create figure with larger, more visible fitness plot
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Main title
    fig.suptitle(f'{csv_file.replace(".csv", "")} - Two-Stage Co-Evolution Results',
                 fontsize=18, fontweight='bold')

    # 1. LARGE Fitness plot - takes full top row with DUAL Y-AXES
    ax1 = fig.add_subplot(gs[0, :])
    marker_freq = max(1, len(df) // 20)  # ~20 markers across the plot

    # Left y-axis for Prey fitness
    line1 = ax1.plot(df['Generation'], df['Prey_Fitness'], label='Prey Fitness',
                     color='#1f77b4', linewidth=2.5, marker='o', markevery=marker_freq,
                     markersize=6, alpha=0.9)

    ax1.set_xlabel('Generation', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Prey Fitness', fontsize=13, fontweight='bold', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=11)

    # Right y-axis for Predator fitness
    ax1_twin = ax1.twinx()
    line2 = ax1_twin.plot(df['Generation'], df['Predator_Fitness'], label='Predator Fitness',
                          color='#d62728', linewidth=2.5, marker='s', markevery=marker_freq,
                          markersize=6, alpha=0.9)

    ax1_twin.set_ylabel('Predator Fitness', fontsize=13, fontweight='bold', color='#d62728')
    ax1_twin.tick_params(axis='y', labelcolor='#d62728', labelsize=11)

    # Stage background shading
    ax1.axvspan(0, stage1_end+0.5, alpha=0.1, color='blue', zorder=0)
    ax1.axvspan(stage1_end+0.5, df['Generation'].max()+1, alpha=0.1, color='red', zorder=0)
    ax1.axvline(x=stage1_end+0.5, color='black', linestyle='--', linewidth=2, alpha=0.7, zorder=1)

    # Annotations
    ax1.annotate(f'Stage 2 Starts\n(Gen {stage1_end+1})',
                xy=(stage1_end+0.5, df['Prey_Fitness'].max()*0.95),
                xytext=(stage1_end+10, df['Prey_Fitness'].max()*0.9),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax1.set_title('Fitness Evolution Over Generations (Dual Y-Axes)', fontsize=15, fontweight='bold', pad=15)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=11, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.4, linestyle='--', zorder=0)
    ax1.tick_params(axis='x', labelsize=11)

    # 2. Energy levels
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df['Generation'], df['Avg_Prey_Energy'], label='Prey Energy',
             color='#1f77b4', linewidth=2, alpha=0.8)
    ax2.plot(df['Generation'], df['Avg_Predator_Energy'], label='Predator Energy',
             color='#d62728', linewidth=2, alpha=0.8)
    ax2.axvline(x=stage1_end+0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Generation', fontsize=11)
    ax2.set_ylabel('Average Energy', fontsize=11)
    ax2.set_title('Energy Levels', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Prey mortality causes
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df['Generation'], df['Prey_Starved'], label='Starved',
             color='orange', linewidth=2, alpha=0.8)
    ax3.plot(df['Generation'], df['Prey_Eaten'], label='Eaten',
             color='purple', linewidth=2, alpha=0.8)
    ax3.axvline(x=stage1_end+0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Generation', fontsize=11)
    ax3.set_ylabel('Average Count', fontsize=11)
    ax3.set_title('Prey Mortality Causes', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Foraging behavior
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(df['Generation'], df['Avg_Food_Collected'], label='Food Collected',
             color='green', linewidth=2, alpha=0.8)
    ax4.axvline(x=stage1_end+0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.set_xlabel('Generation', fontsize=11)
    ax4.set_ylabel('Avg Food Collected', fontsize=11)
    ax4.set_title('Foraging Behavior', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 5. Predator starvation
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(df['Generation'], df['Pred_Starved'], label='Pred Starved',
             color='red', linewidth=2, alpha=0.8)
    ax5.axvline(x=stage1_end+0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax5.set_xlabel('Generation', fontsize=11)
    ax5.set_ylabel('Predators Starved', fontsize=11)
    ax5.set_title('Predator Starvation', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_name = csv_file.replace('.csv', '_plot.png')
    plot_file = os.path.join(output_dir, plot_name)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  → Plot saved to: {plot_file}")

    plt.close()

# Process all CSV files
print("Starting batch plotting...")
print(f"Output directory: {output_dir}/")
print("-" * 60)

for csv_file in csv_files:
    plot_results(csv_file)
    print()

print("-" * 60)
print("Batch plotting complete!")
print(f"All plots saved to: {output_dir}/")
