"""
Inspect checkpoint structure to understand what's available for transfer learning
"""
import torch
import sys
from pathlib import Path

# Check one of the modular addition checkpoints
checkpoint_path = Path('saved_runs/wd_10-1_mod_addition_loss_curve.pth')

print(f"Loading checkpoint from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"\nCheckpoint keys: {checkpoint.keys()}")
print("\n" + "="*80)

# Print details for each key
for key in checkpoint.keys():
    value = checkpoint[key]
    if isinstance(value, list):
        print(f"\n{key}: list of length {len(value)}")
        if len(value) > 0:
            print(f"  First element type: {type(value[0])}")
            if key in ['train_losses', 'test_losses']:
                print(f"  First few values: {value[:5]}")
                print(f"  Last few values: {value[-5:]}")
    elif isinstance(value, dict):
        print(f"\n{key}: dict with keys {list(value.keys())[:10]}...")
        if key == 'config':
            print(f"  Config contents: {value}")
        elif key == 'model':
            print(f"  Model state_dict with {len(value)} parameters")
    else:
        print(f"\n{key}: {type(value)} = {value}")

# If there are test_losses, find when grokking occurred
if 'test_losses' in checkpoint:
    test_losses = checkpoint['test_losses']
    train_losses = checkpoint['train_losses']

    print("\n" + "="*80)
    print("\nAnalyzing grokking progression:")
    print(f"Total epochs: {len(test_losses)}")

    # Find when test accuracy reached 90% (log loss ~ -2.3)
    for i, (train_loss, test_loss) in enumerate(zip(train_losses[-10:], test_losses[-10:])):
        print(f"Epoch {len(test_losses)-10+i}: train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")

    # Check if model is fully grokked
    final_test_loss = test_losses[-1]
    if final_test_loss < 0.01:
        print(f"\n✓ Model is FULLY GROKKED (final test loss: {final_test_loss:.6f})")
    else:
        print(f"\n✗ Model NOT fully grokked (final test loss: {final_test_loss:.6f})")
