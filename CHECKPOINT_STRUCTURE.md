# Checkpoint Structure

## Overview

The training pipeline saves 4-6 checkpoint files at key milestones, focusing on important stages rather than every epoch.

## Checkpoint Files

### 1. Best Model (Always Saved)

**Filename**: `best_model_f1_{score}.pth`

**When saved**: Whenever validation F1 improves during training

**Example**: `best_model_patient_pooling.pth` (F1 score varies by training)

**Contains**:
- Model state dict (all weights)
- Optimizer state dict
- Training epoch
- Validation metrics (F1, sensitivity, specificity)
- Optimal classification threshold
- Complete training history

**Use for**: Final evaluation, deployment, inference

### 2. Stage 1 Checkpoint (Epoch 5)

**Filename**: `checkpoint_epoch05_stage1_f1_{score}.pth`

**When saved**: At the end of Stage 1 (classifier-only training)

**Training state**:
- Classifier trained: ✓
- Layer4 frozen: ✓
- Layer3 frozen: ✓
- Parameters trained: 1.05M (4.3%)

**Use for**: Analyzing classifier-only performance, debugging early training

### 3. Stage 2 Checkpoint (Epoch 12)

**Filename**: `checkpoint_epoch12_stage2_f1_{score}.pth`

**When saved**: At the end of Stage 2 (Layer4 fine-tuning complete)

**Training state**:
- Classifier trained: ✓
- Layer4 fine-tuned: ✓
- Layer3 frozen: ✓
- Parameters trained: 7.9M (32%)

**Use for**: Analyzing Layer4 contribution, resuming partial training

### 4. Final Checkpoint (Epoch 40)

**Filename**: `checkpoint_epoch40_stage3_f1_{score}.pth`

**When saved**: At the end of training (all stages complete)

**Training state**:
- Classifier trained: ✓
- Layer4 fine-tuned: ✓
- Layer3 fine-tuned: ✓
- Parameters trained: 14.9M (61%)

**Use for**: Final model evaluation, comparing with best model

## Checkpoint Contents

Each checkpoint file contains:

```python
{
    'epoch': int,                    # Training epoch (1-40)
    'model_state_dict': OrderedDict, # All model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'val_metrics': {                 # Validation metrics
        'f1': float,
        'sensitivity': float,
        'specificity': float,
        'accuracy': float
    },
    'optimal_threshold': float,      # Best classification threshold
    'history': {                     # Complete training history
        'train_loss': list,
        'val_loss': list,
        'val_f1': list,
        'val_sensitivity': list,
        'val_specificity': list,
        'optimal_threshold': list,
        'separation': list
    }
}
```

## Loading Checkpoints

### Load Best Model for Inference

```python
import torch
from src.models.hybrid_model import ResNet50_ImageOnly

# Load checkpoint
checkpoint = torch.load('YOUR_CHECKPOINT.pth', map_location='cpu')

# Create model
model = ResNet50_ImageOnly(num_classes=1, dropout_rate=0.4)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get optimal threshold
optimal_threshold = checkpoint['optimal_threshold']
print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"Val F1: {checkpoint['val_metrics']['f1']:.4f}")
```

### Resume Training from Checkpoint

```python
# Load checkpoint
checkpoint = torch.load('checkpoint_epoch12_stage2_f1_XXXX.pth')

# Create model and optimizer
model = ResNet50_ImageOnly(num_classes=1, dropout_rate=0.4)
optimizer = torch.optim.Adam(model.parameters())

# Restore states
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
history = checkpoint['history']

print(f"Resuming from epoch {start_epoch}")
```

## File Size

Typical checkpoint file sizes:
- Each checkpoint: ~90-100 MB
- Total for 4 checkpoints: ~360-400 MB

This is much more efficient than saving all 40 epochs (~3.6 GB).

## Checkpoint Selection Strategy

### Why These Specific Epochs?

**Epoch 5 (Stage 1 end)**:
- Marks completion of classifier-only training
- Baseline performance before backbone fine-tuning
- Useful for ablation studies

**Epoch 12 (Stage 2 end)**:
- Layer4 fine-tuning complete
- Often shows significant F1 improvement over Stage 1
- Good checkpoint for partial training analysis

**Epoch 40 (Final)**:
- Complete training with all layers fine-tuned
- May or may not be the best model (depends on overfitting)

**Best Model (variable)**:
- Highest validation F1 achieved during training
- Can occur at any epoch (commonly around epochs 15-30)
- This is the model to use for deployment

## Comparing Checkpoints

To analyze training progression:

```python
import torch

checkpoints = [
    'checkpoint_epoch05_stage1_f1_XXXX.pth',
    'checkpoint_epoch12_stage2_f1_XXXX.pth',
    'checkpoint_epoch40_stage3_f1_XXXX.pth',
    'best_model_patient_pooling.pth'
]

print("Training Progression:")
print("-" * 60)
for ckpt_path in checkpoints:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(f"\n{ckpt_path}")
    print(f"  Epoch: {ckpt['epoch']}")
    print(f"  Val F1: {ckpt['val_metrics']['f1']:.4f}")
    print(f"  Sensitivity: {ckpt['val_metrics']['sensitivity']:.4f}")
    print(f"  Specificity: {ckpt['val_metrics']['specificity']:.4f}")
    print(f"  Threshold: {ckpt['optimal_threshold']:.3f}")
```

## Best Practices

1. **Always use best_model_f1_*.pth for inference and deployment**
2. **Keep Stage 2 checkpoint** for ablation studies
3. **Compare final vs best** to check for late-stage overfitting
4. **Archive old checkpoints** when starting new training runs
5. **Name checkpoints clearly** if running multiple experiments

## Disk Space Management

If disk space is limited:

**Must keep**:
- `best_model_f1_*.pth` (for inference)

**Optional to keep**:
- Stage 1 checkpoint (for analysis)
- Stage 2 checkpoint (for analysis)
- Final checkpoint (for comparison)

**Can delete safely**:
- All other epoch checkpoints (if saved accidentally)
- Checkpoints from failed/incomplete runs

## Summary

The checkpoint strategy saves **4-6 files** instead of 40, focusing on:
- Best performance (best model)
- Key training milestones (stage transitions)
- Final state (full training complete)

This provides sufficient information for analysis while keeping disk usage reasonable.
