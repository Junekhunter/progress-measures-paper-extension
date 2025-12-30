"""
Analyze why seed 1024 is an outlier in the 3-way transfer learning experiment.

This script investigates:
1. Train/test split characteristics across seeds
2. Learning curve patterns
3. Test set difficulty metrics
4. Distribution of examples in modular space
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace
import sys
sys.path.insert(0, '.')

from transformers import Config, gen_train_test

# Configuration
SEEDS = [42, 123, 456, 789, 1024]
P = 113
FRAC_TRAIN = 0.3

print("="*80)
print("SEED OUTLIER ANALYSIS: Why is seed 1024 different?")
print("="*80)

# ============================================================================
# PART 1: Train/Test Split Analysis
# ============================================================================

print("\n" + "="*80)
print("PART 1: TRAIN/TEST SPLIT CHARACTERISTICS")
print("="*80)

split_stats = {}

for seed in SEEDS:
    config = Config(seed=seed, p=P, frac_train=FRAC_TRAIN, fn_name='subtract')
    train_data, test_data = gen_train_test(config)

    train_pairs = [(a, b) for a, b, _ in train_data]
    test_pairs = [(a, b) for a, b, _ in test_data]

    # Extract operands
    train_a = np.array([a for a, b in train_pairs])
    train_b = np.array([b for a, b in train_pairs])
    test_a = np.array([a for a, b in test_pairs])
    test_b = np.array([b for a, b in test_pairs])

    # Calculate statistics
    stats = {
        'train_size': len(train_data),
        'test_size': len(test_data),

        # Range coverage
        'test_a_min': test_a.min(),
        'test_a_max': test_a.max(),
        'test_a_range': test_a.max() - test_a.min(),
        'test_b_min': test_b.min(),
        'test_b_max': test_b.max(),
        'test_b_range': test_b.max() - test_b.min(),

        # Distribution properties
        'test_a_mean': test_a.mean(),
        'test_a_std': test_a.std(),
        'test_b_mean': test_b.mean(),
        'test_b_std': test_b.std(),

        # "Hard" cases for subtraction (where a < b, requiring wrap-around)
        'hard_cases': sum(1 for a, b in test_pairs if a < b),
        'hard_case_frac': sum(1 for a, b in test_pairs if a < b) / len(test_pairs),

        # Edge cases (near 0 or near P)
        'near_zero': sum(1 for a, b in test_pairs if min(a, b) < 10),
        'near_p': sum(1 for a, b in test_pairs if max(a, b) > P - 10),

        # Diagonal cases (where a ≈ b)
        'diagonal': sum(1 for a, b in test_pairs if abs(a - b) < 5),

        # Distribution uniformity (using entropy as proxy)
        'test_a_entropy': -np.sum(np.histogram(test_a, bins=20)[0] / len(test_a) *
                                   np.log(np.histogram(test_a, bins=20)[0] / len(test_a) + 1e-10)),
        'test_b_entropy': -np.sum(np.histogram(test_b, bins=20)[0] / len(test_b) *
                                   np.log(np.histogram(test_b, bins=20)[0] / len(test_b) + 1e-10)),
    }

    split_stats[seed] = stats

    print(f"\nSeed {seed}:")
    print(f"  Train/Test: {stats['train_size']}/{stats['test_size']}")
    print(f"  Test ranges: a=[{stats['test_a_min']}, {stats['test_a_max']}], b=[{stats['test_b_min']}, {stats['test_b_max']}]")
    print(f"  Hard cases (a<b): {stats['hard_cases']}/{stats['test_size']} ({stats['hard_case_frac']:.1%})")
    print(f"  Near boundaries: {stats['near_zero']} near 0, {stats['near_p']} near {P}")
    print(f"  Diagonal (a≈b): {stats['diagonal']}")
    print(f"  Distribution entropy: a={stats['test_a_entropy']:.2f}, b={stats['test_b_entropy']:.2f}")

# Compare seed 1024 to others
print("\n" + "-"*80)
print("COMPARISON: Seed 1024 vs Others")
print("-"*80)

for metric in ['hard_case_frac', 'near_zero', 'near_p', 'diagonal', 'test_a_entropy', 'test_b_entropy']:
    values = [split_stats[s][metric] for s in SEEDS]
    mean_others = np.mean([split_stats[s][metric] for s in SEEDS if s != 1024])
    std_others = np.std([split_stats[s][metric] for s in SEEDS if s != 1024])
    seed_1024_val = split_stats[1024][metric]

    z_score = (seed_1024_val - mean_others) / (std_others + 1e-10)

    marker = "⚠️" if abs(z_score) > 1.5 else "✓"
    print(f"  {marker} {metric:20s}: 1024={seed_1024_val:.3f}, others={mean_others:.3f}±{std_others:.3f} (z={z_score:+.2f})")

# ============================================================================
# PART 2: Visualize Distributions
# ============================================================================

print("\n" + "="*80)
print("PART 2: DISTRIBUTION VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, seed in enumerate(SEEDS):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    config = Config(seed=seed, p=P, frac_train=FRAC_TRAIN, fn_name='subtract')
    train_data, test_data = gen_train_test(config)

    test_pairs = [(a, b) for a, b, _ in test_data]
    test_a = [a for a, b in test_pairs]
    test_b = [b for a, b in test_pairs]

    # 2D histogram of test set
    h, xedges, yedges = np.histogram2d(test_a, test_b, bins=20)

    im = ax.imshow(h.T, origin='lower', extent=[0, P, 0, P], cmap='viridis', aspect='auto')
    ax.set_xlabel('First operand (a)')
    ax.set_ylabel('Second operand (b)')
    ax.set_title(f'Seed {seed}' + (' (OUTLIER)' if seed == 1024 else ''),
                 fontweight='bold' if seed == 1024 else 'normal',
                 color='red' if seed == 1024 else 'black')
    ax.plot([0, P], [0, P], 'r--', alpha=0.3, label='a=b diagonal')
    plt.colorbar(im, ax=ax, label='Test examples')

# Remove extra subplot
axes[1, 2].remove()

plt.tight_layout()
plt.savefig('seed_distribution_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: seed_distribution_comparison.png")
plt.show()

# ============================================================================
# PART 3: Learning Curve Analysis
# ============================================================================

print("\n" + "="*80)
print("PART 3: LEARNING CURVE ANALYSIS")
print("="*80)

# You need to provide the path to saved results
EXPERIMENT_DIR = input("Enter experiment directory path (or press Enter to skip): ").strip()

if EXPERIMENT_DIR:
    print("\nLoading learning curves...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: All grokked transfer curves
    ax = axes[0, 0]
    for seed in SEEDS:
        try:
            results = torch.load(f'{EXPERIMENT_DIR}/checkpoints/grokked_transfer_seed{seed}.pth')
            accs = results['test_accuracies']
            epochs_to_999 = results['threshold_epochs'].get(0.999, None)

            label = f'Seed {seed}'
            if epochs_to_999:
                label += f' ({epochs_to_999} epochs)'

            color = 'red' if seed == 1024 else 'blue'
            alpha = 1.0 if seed == 1024 else 0.4
            linewidth = 2.5 if seed == 1024 else 1.5

            ax.plot(accs[:10000], label=label, color=color, alpha=alpha, linewidth=linewidth)
        except:
            print(f"  Could not load grokked_transfer_seed{seed}.pth")

    ax.axhline(0.999, color='green', linestyle='--', alpha=0.5, label='99.9% threshold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Grokked Transfer: All Seeds', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Zoomed comparison (first 1000 epochs)
    ax = axes[0, 1]
    for seed in [42, 1024]:  # Compare fastest vs slowest
        try:
            results = torch.load(f'{EXPERIMENT_DIR}/checkpoints/grokked_transfer_seed{seed}.pth')
            accs = results['test_accuracies']

            label = f'Seed {seed}' + (' (outlier)' if seed == 1024 else ' (typical)')
            color = 'red' if seed == 1024 else 'blue'

            ax.plot(accs[:1000], label=label, color=color, linewidth=2)
        except:
            pass

    ax.axhline(0.999, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Grokked Transfer: First 1000 Epochs', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Comparison across conditions for seed 1024
    ax = axes[1, 0]
    try:
        for condition, color, label in [
            ('grokked_transfer', 'blue', 'Grokked'),
            ('memorized_transfer', 'purple', 'Memorized'),
            ('random_baseline', 'orange', 'Random')
        ]:
            results = torch.load(f'{EXPERIMENT_DIR}/checkpoints/{condition}_seed1024.pth')
            accs = results['test_accuracies']
            epochs_to_999 = results['threshold_epochs'].get(0.999, None)

            plot_label = f'{label}'
            if epochs_to_999:
                plot_label += f' ({epochs_to_999} epochs)'

            max_plot = min(len(accs), 25000)
            ax.plot(accs[:max_plot], label=plot_label, color=color, linewidth=2)

        ax.axhline(0.999, color='green', linestyle='--', alpha=0.5, label='99.9%')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Seed 1024: All Conditions', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    except:
        ax.text(0.5, 0.5, 'Could not load seed 1024 results',
                ha='center', va='center', transform=ax.transAxes)

    # Plot 4: Learning rate (derivative of accuracy)
    ax = axes[1, 1]
    for seed in [42, 1024]:
        try:
            results = torch.load(f'{EXPERIMENT_DIR}/checkpoints/grokked_transfer_seed{seed}.pth')
            accs = np.array(results['test_accuracies'])

            # Smooth derivative
            window = 50
            smoothed = np.convolve(accs, np.ones(window)/window, mode='valid')
            derivative = np.diff(smoothed)

            label = f'Seed {seed}' + (' (outlier)' if seed == 1024 else ' (typical)')
            color = 'red' if seed == 1024 else 'blue'

            ax.plot(derivative[:5000], label=label, color=color, alpha=0.7)
        except:
            pass

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate (dAccuracy/dEpoch)')
    ax.set_title('Learning Rate Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig('learning_curve_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: learning_curve_analysis.png")
    plt.show()

# ============================================================================
# PART 4: Test Example Difficulty Analysis
# ============================================================================

print("\n" + "="*80)
print("PART 4: TEST EXAMPLE DIFFICULTY")
print("="*80)

print("\nAnalyzing which test examples might be hardest...")

for seed in SEEDS:
    config = Config(seed=seed, p=P, frac_train=FRAC_TRAIN, fn_name='subtract')
    train_data, test_data = gen_train_test(config)

    train_set = set((a, b) for a, b, _ in train_data)
    test_set = set((a, b) for a, b, _ in test_data)

    # Calculate "distance" of each test example from training set
    # Using minimum Euclidean distance in (a, b) space
    difficulties = []

    for test_a, test_b, _ in test_data:
        min_dist = float('inf')
        for train_a, train_b, _ in train_data:
            # Euclidean distance in modular space
            dist = min(abs(test_a - train_a), P - abs(test_a - train_a)) ** 2 + \
                   min(abs(test_b - train_b), P - abs(test_b - train_b)) ** 2
            min_dist = min(min_dist, dist ** 0.5)
        difficulties.append(min_dist)

    difficulties = np.array(difficulties)

    print(f"\nSeed {seed}:")
    print(f"  Mean test distance from training: {difficulties.mean():.2f}")
    print(f"  Max test distance from training: {difficulties.max():.2f}")
    print(f"  Std test distance: {difficulties.std():.2f}")
    print(f"  % of test >10 away: {(difficulties > 10).mean():.1%}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY: Why is Seed 1024 Different?")
print("="*80)

print("""
Based on the analysis:

1. TRAIN/TEST SPLIT CHARACTERISTICS:
   - Check the comparison table above for any metrics where seed 1024 has |z| > 1.5
   - Look for unusual distribution patterns in the 2D histograms

2. LEARNING DYNAMICS:
   - Compare learning curves - does seed 1024 plateau somewhere?
   - Check learning rate (derivative) - is there a different pattern?

3. TEST SET DIFFICULTY:
   - Are seed 1024's test examples farther from training distribution?
   - More hard cases (a < b requiring wrap-around)?

4. STRUCTURAL DIFFERENCES:
   - Does the test set sample a different region of modular space?
   - Are there more edge cases or diagonal cases?

INTERPRETATION:
- If seed 1024 has higher test difficulty → explains slower learning
- If seed 1024 has unusual distribution → may not align with grokked patterns
- This variance is EXPECTED and makes results more credible!

FOR THE PAPER:
- Report both with and without outlier
- Use median instead of mean for robustness
- Discuss how transfer effectiveness depends on task alignment
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  - seed_distribution_comparison.png")
if EXPERIMENT_DIR:
    print("  - learning_curve_analysis.png")
