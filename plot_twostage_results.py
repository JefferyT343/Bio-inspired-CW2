"""
Plot results from TwoStageEvolution simulation.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
results_dir = "results"
csv_file = os.path.join(results_dir, "TwoStageEvolution_100gens.csv")

if not os.path.exists(csv_file):
    print(f"Error: Could not find {csv_file}")
    print("Run the simulation first to generate results.")
    exit(1)

# Read data
df = pd.read_csv(csv_file)

# Identify stage transition
stage1_end = df[df['Stage'] == 1]['Generation'].max()

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Two-Stage Co-Evolution Results (Stage 1: Prey Only → Stage 2: Co-evolution)',
             fontsize=16, fontweight='bold')

# 1. Fitness over generations with stage marker
ax1 = axes[0, 0]
ax1.plot(df['Generation'], df['Prey_Fitness'], label='Prey', color='blue', linewidth=2)
ax1.plot(df['Generation'], df['Predator_Fitness'], label='Predator', color='red', linewidth=2)
ax1.axvline(x=stage1_end, color='gray', linestyle='--', linewidth=2, alpha=0.7, label=f'Stage 2 Start (Gen {stage1_end+1})')
ax1.set_xlabel('Generation')
ax1.set_ylabel('Fitness')
ax1.set_title('Fitness Evolution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Energy levels
ax2 = axes[0, 1]
ax2.plot(df['Generation'], df['Avg_Prey_Energy'], label='Prey Energy', color='blue', linewidth=2)
ax2.plot(df['Generation'], df['Avg_Predator_Energy'], label='Predator Energy', color='red', linewidth=2)
ax2.axvline(x=stage1_end, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax2.set_xlabel('Generation')
ax2.set_ylabel('Average Energy')
ax2.set_title('Energy Levels')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Prey mortality causes
ax3 = axes[1, 0]
ax3.plot(df['Generation'], df['Prey_Starved'], label='Starved', color='orange', linewidth=2)
ax3.plot(df['Generation'], df['Prey_Eaten'], label='Eaten', color='purple', linewidth=2)
ax3.axvline(x=stage1_end, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax3.set_xlabel('Generation')
ax3.set_ylabel('Average Count')
ax3.set_title('Prey Mortality Causes')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Foraging behavior
ax4 = axes[1, 1]
ax4.plot(df['Generation'], df['Avg_Food_Collected'], label='Food Collected', color='green', linewidth=2)
ax4_twin = ax4.twinx()
ax4_twin.plot(df['Generation'], df['Pred_Starved'], label='Pred Starved', color='red',
              linewidth=2, linestyle='--', alpha=0.7)
ax4.axvline(x=stage1_end, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax4.set_xlabel('Generation')
ax4.set_ylabel('Avg Food Collected', color='green')
ax4_twin.set_ylabel('Pred Starved', color='red')
ax4.set_title('Foraging & Predator Starvation')
ax4.tick_params(axis='y', labelcolor='green')
ax4_twin.tick_params(axis='y', labelcolor='red')
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save plot
plot_file = os.path.join(results_dir, "TwoStageEvolution_100gens_plot.png")
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_file}")

# Show plot
plt.show()
