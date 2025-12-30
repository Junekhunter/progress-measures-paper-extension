# Transfer Learning Experiment: Grokked Addition → Subtraction

## Overview

This experiment tests whether a **grokked modular addition model** can transfer to **accelerate learning on modular subtraction**.

The key question: **Does transfer from grokked addition → subtraction reduce the number of training epochs needed compared to random initialization?**

## Experiment Design

### Setup
- **Task**: Modular arithmetic (mod 113)
- **Source task**: Addition (a + b mod 113)
- **Target task**: Subtraction (a - b mod 113)
- **Source model**: Fully grokked addition model from `saved_runs/wd_10-1_mod_addition_loss_curve.pth`

### Two Conditions

1. **Transfer Learning**
   - Initialize subtraction model with weights from grokked addition model
   - Fine-tune on subtraction task

2. **Baseline**
   - Initialize subtraction model with random weights
   - Train from scratch on subtraction task

### Key Metrics

- **Epochs to 90% test accuracy**: How many epochs to reach 90% accuracy?
- **Test accuracy curves**: How does accuracy improve over training?
- **Training loss curves**: How does loss decrease over training?

## Running the Experiment

We provide **two notebooks** for different use cases:

### Option 1: Quick Single-Run (`transfer_learning_experiment.ipynb`)
- **Best for**: Quick exploration, testing the setup
- **Runtime**: ~20-40 minutes on GPU
- **Direct link**: [Open in Colab](https://colab.research.google.com/github/Junekhunter/progress-measures-paper-extension/blob/main/transfer_learning_experiment.ipynb)

### Option 2: Multi-Seed Robust Analysis (`transfer_learning_multiseed.ipynb`) ⭐ **RECOMMENDED**
- **Best for**: Publication-quality results with statistical analysis
- **Features**:
  - ✅ Runs 5 different seeds for statistical robustness
  - ✅ Google Drive integration for persistent storage
  - ✅ Aggregated results with mean/std/confidence intervals
  - ✅ Publication-ready visualizations
- **Runtime**: ~2-3 hours on GPU (5 seeds × 2 conditions)
- **Direct link**: [Open in Colab](https://colab.research.google.com/github/Junekhunter/progress-measures-paper-extension/blob/main/transfer_learning_multiseed.ipynb)

### Quick Start (Multi-Seed)

1. **Open the multi-seed notebook in Colab** using the link above
2. **Mount Google Drive** when prompted (results will be saved there)
3. **Run all cells** - the notebook handles everything:
   - Clones repository and installs dependencies
   - Loads grokked addition checkpoint
   - Runs 5×2=10 experiments (5 seeds, 2 conditions each)
   - Aggregates results with statistical analysis
   - Generates publication-quality figures
   - Saves everything to Google Drive
4. **Results saved to**: `MyDrive/grokking_transfer_experiments/run_TIMESTAMP/`

### Locally (if dependencies installed)

```bash
# Clone the repository
git clone https://github.com/Junekhunter/progress-measures-paper-extension.git
cd progress-measures-paper-extension

# Install dependencies
pip install torch numpy einops matplotlib tqdm

# Open and run the Jupyter notebook
jupyter notebook transfer_learning_experiment.ipynb
```

## Expected Results

### Hypothesis
If the grokked addition model has learned generalizable representations of modular arithmetic, transfer learning should:
- Reach 90% accuracy **faster** than random initialization
- Show **lower initial loss** (better starting point)
- Potentially achieve **better final accuracy**

### Alternative Outcomes

1. **Strong transfer**: Transfer model reaches 90% in <50% of the epochs needed by baseline
2. **Weak transfer**: Transfer model reaches 90% slightly faster, but not dramatically
3. **No transfer**: Both models take similar time to reach 90%
4. **Negative transfer**: Transfer model is slower (unlikely but possible if addition/subtraction representations conflict)

## Files

- **`transfer_learning_experiment.ipynb`**: Single-run experiment notebook
- **`transfer_learning_multiseed.ipynb`**: ⭐ Multi-seed experiment with statistical analysis (recommended)
- **`transformers.py`**: Model architecture (1-layer transformer)
- **`helpers.py`**: Helper functions for training and evaluation
- **`saved_runs/`**: Pre-trained checkpoints including grokked addition models

## Experiment Parameters

Default settings (can be modified in notebook):
- **Training epochs**: 5,000 per experiment
- **Learning rate**: 1e-3
- **Weight decay**: 1.0
- **Batch style**: Full batch (all training data per step)
- **Train/test split**: 30% train, 70% test
- **Prime modulus**: 113

## Visualization

The notebook generates comprehensive plots:
1. Training loss over time (linear and log scale)
2. Test loss over time (linear and log scale)
3. Test accuracy over time
4. Zoomed view of first 1000 epochs
5. Vertical lines marking when each model reaches 90% accuracy

## Output Files

### Single-Run Notebook
After running `transfer_learning_experiment.ipynb`:
- **`transfer_learning_results.png`**: Visualization of all metrics
- **`transfer_learning_experiment_results.pth`**: PyTorch checkpoint with full results
- **`transfer_learning_experiment_results.npz`**: NumPy arrays for easy analysis

### Multi-Seed Notebook ⭐
After running `transfer_learning_multiseed.ipynb`, results are saved to Google Drive:
```
MyDrive/grokking_transfer_experiments/run_YYYYMMDD_HHMMSS/
├── figures/
│   └── multiseed_results.png          # Publication-quality 6-panel figure
├── checkpoints/
│   ├── transfer_seed42.pth            # Individual model checkpoints
│   ├── transfer_seed123.pth           # (one per seed)
│   ├── baseline_seed42.pth
│   └── ... (10 total)
└── results/
    ├── complete_results.pth           # All experimental data
    ├── curves.npz                     # Numpy arrays of all curves
    └── aggregated_stats.json          # Summary statistics (mean/std/CI)
```

The multi-seed results include:
- Mean and standard deviation across seeds
- 95% confidence intervals
- Individual run traces
- Box plots of key metrics
- Statistical comparison between transfer and baseline

## Citation

This experiment builds on:
- Original paper: "Progress measures for grokking via mechanistic interpretability" (https://github.com/mechanistic-interpretability-grokking/progress-measures-paper)
- Fork with extensions: https://github.com/Junekhunter/progress-measures-paper-extension

## Next Steps

After running the base experiment, you can extend it:

1. **Different source checkpoints**: Try other addition models (e.g., `wd_10-2_mod_addition_loss_curve.pth`)
2. **Different target tasks**: Test transfer to multiplication or other operations
3. **Partially grokked models**: Use checkpoints from earlier in training (before full grokking)
4. **Hyperparameter sweep**: Vary learning rate, weight decay, or training fraction
5. **Layer-wise analysis**: Freeze different layers to see which parts transfer best
6. **Representation analysis**: Visualize activations to understand what transfers

## Questions?

If you encounter issues or have questions about the experiment:
1. Check that the checkpoint file exists in `saved_runs/`
2. Verify CUDA/GPU is available for faster training
3. Try reducing `num_epochs` to 1000 for a quick test run
4. Open an issue on GitHub with your error message
