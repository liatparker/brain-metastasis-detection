# Model Architecture

## Overview

The brain metastasis detection system uses a ResNet-50 backbone pretrained on RadImageNet, adapted for single-channel CT imaging with patient-level prediction aggregation.

## Network Architecture

### ResNet50-ImageOnly

```
Input: CT Slice
    [1, 256, 256]
         ↓
    ┌─────────────────────┐
    │   Conv1 + BN + ReLU │  [64, 128, 128]
    │      MaxPool         │  [64, 64, 64]
    └─────────────────────┘
         ↓
    ┌─────────────────────┐
    │      Layer 1         │  [256, 64, 64]
    │   (3 blocks)         │
    └─────────────────────┘
         ↓
    ┌─────────────────────┐
    │      Layer 2         │  [512, 32, 32]
    │   (4 blocks)         │
    └─────────────────────┘
         ↓
    ┌─────────────────────┐
    │      Layer 3         │  [1024, 16, 16]
    │   (6 blocks)         │
    └─────────────────────┘
         ↓
    ┌─────────────────────┐
    │      Layer 4         │  [2048, 8, 8]
    │   (3 blocks)         │
    └─────────────────────┘
         ↓
    ┌─────────────────────┐
    │  Global Avg Pool     │  [2048]
    └─────────────────────┘
         ↓
    ┌─────────────────────┐
    │   Linear(2048→512)   │
    │       ReLU           │
    │    Dropout(0.4)      │
    │    Linear(512→1)     │
    └─────────────────────┘
         ↓
    Output: Logit
```

### Layer Specifications

**Backbone (ResNet-50):**
- Total parameters: 23.5M
- Pretrained on RadImageNet (medical images)
- Modified for single-channel input
- Uses Batch Normalization throughout

**Classifier:**
- Hidden layer: 512 units
- Dropout: 0.4 (regularization)
- Output: Single logit (binary classification)
- Total parameters: 1.05M

## Patient-Level Prediction Pipeline

Individual CT slices are aggregated to produce a single patient-level prediction:

```
Patient with N slices
    ↓
┌───────────────────────────┐
│  Slice 1 → Model → logit₁ │
│  Slice 2 → Model → logit₂ │
│     ...                    │
│  Slice N → Model → logitₙ │
└───────────────────────────┘
    ↓
MAX Pooling
    ↓
patient_logit = max(logit₁, logit₂, ..., logitₙ)
    ↓
Loss = BCE(patient_logit, patient_label)
```

### Why MAX Pooling?

- **Clinical reasoning**: If ANY slice shows metastasis, patient is positive
- **Small lesion detection**: Sensitive to lesions appearing in few slices
- **Robust**: Not diluted by many negative slices

## Training Strategy

### Curriculum Learning (3 Stages)

Training progresses through three stages with automatic layer unfreezing:

**Stage 1 (Epochs 0-4):**
```
Frozen: conv1, layer1, layer2, layer3, layer4
Trainable: Classifier only
Parameters: 1.05M (4.3%)
Learning Rate: 5e-5
```

Purpose: Classifier learns to use frozen RadImageNet features

**Stage 2 (Epochs 5-11):**
```
Frozen: conv1, layer1, layer2, layer3
Trainable: Layer4 + Classifier
Parameters: 7.9M (32%)
Learning Rates:
  - Layer4: 1e-5 (careful adaptation)
  - Classifier: 5e-5 (faster adaptation)
```

Purpose: Adapt high-level features while classifier stabilizes

**Stage 3 (Epochs 12-39):**
```
Frozen: conv1, layer1, layer2
Trainable: Layer3 + Layer4 + Classifier
Parameters: 14.9M (61%)
Learning Rates:
  - Layer3: 5e-6 (most conservative)
  - Layer4: 1e-5 (medium)
  - Classifier: 5e-5 (most aggressive)
```

Purpose: Full model adaptation with differential learning rates

### Rationale for Differential Learning Rates

Lower layers encode general image features (edges, textures) that transfer well from RadImageNet. Higher layers and the classifier are task-specific and benefit from faster adaptation.

**Learning rate ratio:**
- Classifier : Layer4 : Layer3 = 10 : 2 : 1

This prevents catastrophic forgetting of pretrained features while enabling task-specific learning.

## Loss Function

### Base Loss
Binary Cross-Entropy with Logits (BCEWithLogitsLoss):
```
L_base = -[y·log(σ(z)) + (1-y)·log(1-σ(z))]
```

Where:
- y: true label (0 or 1)
- z: model output logit
- σ: sigmoid function

### Confidence Optimization (Optional)

Adds a penalty term to encourage separation between positive and negative predictions:

```
L_confidence = (max(0, T - (mean(pred_pos) - mean(pred_neg))))²
L_total = L_base + λ·L_confidence
```

Where:
- T: target separation (default 0.30)
- λ: confidence weight (default 0.15)
- Penalty is only applied when separation < T

**Purpose**: Increases model confidence by explicitly optimizing for larger gaps between positive and negative predictions.

## Regularization Techniques

### 1. Differential Weight Decay
- Backbone layers: 1e-4 (light regularization, preserve pretrained features)
- Classifier: 1e-3 (strong regularization, prevent overfitting)

### 2. Dropout
- Rate: 0.4
- Location: Between classifier hidden layers
- Purpose: Prevents overfitting on small positive class

### 3. Gradient Clipping
- Max norm: 1.0
- Purpose: Prevents gradient explosion and training instability

### 4. Class-Conditional Augmentation
See PREPROCESSING.md for details on differential augmentation strategies.

## Data Handling

### Patient-Level Sampling

Custom PyTorch sampler ensures:
- No patient appears in both train and validation sets
- Balanced positive:negative ratio (1:1.5) in each batch
- Random patient ordering each epoch

### Custom Collate Function

Groups all slices from the same patient into a single batch item:

```python
patient_collate_fn(batch):
    # Groups slices by patient_id
    # Returns list of (images, features, label, patient_id) per patient
```

This enables patient-level loss computation and prevents slice-level bias.

## Key Implementation Details

### Input Preprocessing
1. DICOM → Hounsfield Units conversion
2. Brain window: Center=40 HU, Width=80 HU
3. Z-score normalization
4. Resize to 256×256
5. Convert to single-channel tensor [1, 256, 256]

### Single-Channel Adaptation
ResNet-50 expects 3-channel input. We adapt the first convolutional layer:
```python
# Average RGB weights to create single-channel filter
conv1_single = conv1_rgb.mean(dim=1, keepdim=True)
```

This preserves learned edge detection from pretraining.

### Batch Normalization
- Present in all backbone layers
- Frozen in layers 0-6 during training
- Active in layer 7 (layer4) from epoch 5 onwards
- Active in layer 6 (layer3) from epoch 12 onwards

## Inference

### Single Patient Prediction

```python
model.eval()
with torch.no_grad():
    slice_logits = []
    for slice_image in patient_slices:
        logit = model(slice_image)
        slice_logits.append(logit)
    
    patient_logit = torch.stack(slice_logits).max()
    patient_prob = torch.sigmoid(patient_logit)
    prediction = (patient_prob > optimal_threshold).int()
```

### Optimal Threshold

The optimal classification threshold is determined by:
1. Sweeping thresholds from 0.05 to 0.95
2. Computing F1 score at each threshold
3. Selecting threshold that maximizes F1

Typical optimal thresholds: 0.15-0.35 (lower than standard 0.5 due to class imbalance)

## Performance Characteristics

### Typical Learning Curve

```
Epoch 0-4 (Stage 1):
  Val F1: 0.45 → 0.58
  Mean pred POS: 0.30-0.40, NEG: 0.25-0.35

Epoch 5-11 (Stage 2):
  Val F1: 0.58 → 0.65
  Mean pred POS: 0.40-0.50, NEG: 0.30-0.35

Epoch 12-39 (Stage 3):
  Val F1: 0.65 → 0.75
  Mean pred POS: 0.50-0.60, NEG: 0.30-0.40
```

### Confidence Metrics

**Separation** = mean(positive predictions) - mean(negative predictions)

Good separation indicates confident, well-calibrated predictions:
- Poor separation: < 0.10
- Moderate separation: 0.10-0.20
- Good separation: 0.20-0.30
- Excellent separation: > 0.30

## Troubleshooting

### Model Collapse (Predictions → 0)
**Symptoms**: Mean predictions approaching 0, all metrics = 0
**Solutions**:
- Reduce learning rates
- Freeze more layers
- Check gradient clipping is active
- Verify patient-level loss scaling

### No Learning on Positives (Sensitivity = 0)
**Symptoms**: Model predicts all negatives
**Solutions**:
- Check class balance in batches
- Increase pos_weight in loss function
- Reduce augmentation intensity on positives
- Verify patient-level sampling is working

### Training Instability (Loss oscillating)
**Symptoms**: Validation loss jumping wildly
**Solutions**:
- Reduce learning rates
- Increase gradient clipping (reduce max_norm)
- Freeze more backbone layers
- Check BatchNorm is in eval mode during validation

## References

1. RadImageNet: Open medical image dataset for transfer learning
2. ResNet-50: Deep Residual Learning for Image Recognition (He et al., 2016)
3. Curriculum Learning strategies for medical imaging
4. Class-conditional augmentation for imbalanced datasets
