# Patient-Level Training

## Overview

This document explains the patient-level training strategy, including patient-level sampling, loss computation, and prediction pooling.

## Why Patient-Level Training?

### Problem with Slice-Level Training

Traditional slice-level training has critical flaws for medical imaging:

1. **Data Leakage**: Multiple slices from the same patient can appear in both training and validation sets
2. **Biased Loss**: Patients with more slices dominate the loss function
3. **Unrealistic Evaluation**: A patient with 50 slices counts as 50 samples, skewing metrics
4. **Clinical Mismatch**: Diagnosis is per-patient, not per-slice

### Patient-Level Solution

Patient-level training ensures:
- **No Data Leakage**: Entire patients are assigned to train OR validation
- **Fair Loss**: Each patient contributes equally to loss, regardless of slice count
- **Clinical Alignment**: Evaluation matches real-world diagnosis workflow
- **Better Generalization**: Model learns patient-level patterns

## Components

### 1. Patient-Level Data Splitting

**Implementation**: `src/preprocessing/data_balancing.py`

Splits data at the patient level, not slice level:

```python
from src.preprocessing.data_balancing import create_patient_level_split

train_df, val_df = create_patient_level_split(
    labels_df,
    train_ratio=0.8,
    random_seed=42
)
```

**Key Features**:
- Splits patients (not slices) into 80% train, 20% validation
- Stratified split maintains class balance
- No patient appears in both sets
- Ensures true generalization to unseen patients

**Output**:
```
Train: 80% of patients → all their slices
Val:   20% of patients → all their slices
```

### 2. PatientSampler

**Implementation**: `src/data/dataset.py` - `PatientSampler` class

Custom PyTorch sampler that groups slices by patient for batching.

**Purpose**:
- Each batch contains all slices from ONE patient
- Maintains class balance (1.5:1 negative:positive patient ratio)
- Enables patient-level loss computation

**Usage**:
```python
from src.data.dataset import PatientSampler

train_dataset = BrainMetDataset(train_df, ...)
sampler = PatientSampler(train_dataset, shuffle=True)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=sampler,
    collate_fn=patient_collate_fn
)
```

**How It Works**:

1. **Groups slices by patient**:
   ```
   patient_001: [slice_001, slice_002, slice_003]
   patient_002: [slice_001, slice_002]
   patient_003: [slice_001, slice_002, slice_003, slice_004]
   ```

2. **Samples patients (not slices)**:
   - Each iteration yields ONE patient
   - All slices from that patient go in the batch

3. **Maintains class balance**:
   - Ratio: 1.5 negative patients : 1 positive patient
   - Example: If 10 positive patients → sample 15 negative patients
   - Prevents class imbalance from dominating training

4. **Shuffle control**:
   - `shuffle=True`: Randomize patient order each epoch
   - `shuffle=False`: Fixed patient order (for validation)

**Batch Structure**:
```
Batch 1: All slices from patient_001 (e.g., 12 slices)
Batch 2: All slices from patient_002 (e.g., 8 slices)
Batch 3: All slices from patient_003 (e.g., 15 slices)
...
```

### 3. patient_collate_fn

**Implementation**: `src/data/dataset.py` - `patient_collate_fn` function

Custom collate function that groups patient slices together.

**Purpose**:
- Converts list of individual slices into patient-grouped structure
- Preserves patient ID for tracking
- Maintains all slices from same patient together

**Input** (from DataLoader):
```python
[
    (img1, label1, 'patient_A'),  # Slice 1 of patient A
    (img2, label1, 'patient_A'),  # Slice 2 of patient A
    (img3, label1, 'patient_A'),  # Slice 3 of patient A
]
```

**Output** (patient-grouped):
```python
[
    [
        (
            torch.stack([img1, img2, img3]),  # All images stacked
            label1,  # Patient-level label
            'patient_A'  # Patient ID
        )
    ]
]
```

**Usage in Training Loop**:
```python
for patient_batches in train_loader:
    for images, label, patient_id in patient_batches:
        # images shape: [num_slices, 1, 256, 256]
        # label: patient-level label (same for all slices)
        # patient_id: unique patient identifier
        
        # Process each slice through model
        slice_logits = []
        for img in images:
            logit = model(img.unsqueeze(0))
            slice_logits.append(logit)
        
        # Pool to patient level
        patient_logit = torch.stack(slice_logits).max()
        
        # Compute loss once per patient
        loss = criterion(patient_logit, label)
```

### 4. Patient-Level Pooling

**Implementation**: `scripts/train.py` - `pool_slice_predictions` function

Aggregates slice-level predictions to a single patient-level prediction.

**Why MAX Pooling?**

```
Clinical Reasoning:
- If ANY slice shows metastasis → patient has metastasis
- Small lesions may appear in only 1-2 slices
- MAX pooling is sensitive to rare positive findings

Mathematical Reasoning:
- patient_logit = max(logit₁, logit₂, ..., logitₙ)
- Takes the MOST confident slice prediction
- Not diluted by many negative slices
```

**Alternative: MEAN Pooling**

```python
patient_logit = torch.stack(slice_logits).mean()
```

**Comparison**:
| Method | Sensitivity | Specificity | Use Case |
|--------|------------|-------------|----------|
| MAX    | Higher     | Lower       | Rare lesions, small metastases |
| MEAN   | Lower      | Higher      | Diffuse pathology, averaging noise |

**For brain metastasis detection**: MAX pooling is preferred because small focal lesions must not be missed.

### 5. Patient-Level Loss Computation

**Key Principle**: Loss computed ONCE per patient, not per slice.

**Wrong (Slice-Level)**:
```python
for images, labels in train_loader:
    outputs = model(images)  # Process all slices
    loss = criterion(outputs, labels)  # Loss per slice
    loss.backward()
```

**Correct (Patient-Level)**:
```python
for patient_batches in train_loader:
    for images, label, patient_id in patient_batches:
        # Process all slices
        slice_logits = []
        for img in images:
            logit = model(img.unsqueeze(0))
            slice_logits.append(logit)
        
        # Pool to patient level
        patient_logit = torch.stack(slice_logits).max()
        
        # Loss computed ONCE per patient
        loss = criterion(patient_logit.unsqueeze(0), label)
        loss.backward()
```

**Loss Scaling**:
```python
# CORRECT: Divide by number of patients
train_loss = total_loss / len(all_patient_preds)

# WRONG: Divide by number of slices or batches
train_loss = total_loss / len(train_loader)  # NO!
train_loss = total_loss / num_slices         # NO!
```

## Complete Training Flow

### 1. Data Preparation

```python
# Load labels
labels_df = pd.read_csv('labels.csv')

# Split at patient level
train_df, val_df = create_patient_level_split(
    labels_df, train_ratio=0.8
)

print(f"Train patients: {train_df['PatientID'].nunique()}")
print(f"Val patients: {val_df['PatientID'].nunique()}")
```

### 2. Create Datasets and Samplers

```python
# Create datasets
train_dataset = BrainMetDataset(train_df, dicom_root, augment=True)
val_dataset = BrainMetDataset(val_df, dicom_root, augment=False)

# Create patient-level samplers
train_sampler = PatientSampler(train_dataset, shuffle=True)
val_sampler = PatientSampler(val_dataset, shuffle=False)

# Create dataloaders with custom collate
train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    collate_fn=patient_collate_fn,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_sampler,
    collate_fn=patient_collate_fn,
    num_workers=0
)
```

### 3. Training Loop

```python
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    all_patient_preds = []
    all_patient_labels = []
    
    for patient_batches in train_loader:
        optimizer.zero_grad()
        
        for images, label, patient_id in patient_batches:
            images = images.to(device)
            label = label.to(device)
            
            # Forward pass for all slices
            slice_logits = []
            for img in images:
                logit = model(img.unsqueeze(0))
                slice_logits.append(logit.squeeze())
            
            # Pool to patient level (MAX)
            patient_logit = torch.stack(slice_logits).max()
            
            # Compute loss ONCE per patient
            loss = criterion(patient_logit.unsqueeze(0), label)
            loss.backward()
            
            # Store predictions
            all_patient_preds.append(patient_logit.detach().cpu().item())
            all_patient_labels.append(label.cpu().item())
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    # Average loss per PATIENT (not per slice!)
    train_loss = total_loss / len(all_patient_preds)
```

### 4. Validation Loop

```python
model.eval()
val_patient_preds = []
val_patient_labels = []

with torch.no_grad():
    for patient_batches in val_loader:
        for images, label, patient_id in patient_batches:
            images = images.to(device)
            label = label.to(device)
            
            # Forward pass
            slice_logits = []
            for img in images:
                logit = model(img.unsqueeze(0))
                slice_logits.append(logit.squeeze())
            
            # Pool to patient level
            patient_logit = torch.stack(slice_logits).max()
            
            # Store predictions
            val_patient_preds.append(patient_logit.cpu().item())
            val_patient_labels.append(label.cpu().item())

# Compute metrics at patient level
val_metrics = compute_metrics(
    np.array(val_patient_preds),
    np.array(val_patient_labels)
)
```

## Benefits

### 1. No Data Leakage
- Train and validation are truly independent
- Model generalizes to unseen patients

### 2. Fair Loss Contribution
- Each patient contributes equally
- Not dominated by patients with many slices

### 3. Clinical Alignment
- One prediction per patient
- Matches real diagnostic workflow

### 4. Robust Metrics
- F1, sensitivity, specificity computed per patient
- Reflects real-world performance

## Common Pitfalls

### Pitfall 1: Slice-Level Split
```python
# WRONG: Splits slices, not patients
train_df, val_df = train_test_split(labels_df, test_size=0.2)
```

**Problem**: Same patient can appear in both train and val.

**Solution**: Use `create_patient_level_split`.

### Pitfall 2: Slice-Level Loss
```python
# WRONG: Loss per slice
loss = criterion(model(images), labels)
```

**Problem**: Patients with more slices dominate loss.

**Solution**: Pool to patient level first, then compute loss.

### Pitfall 3: Wrong Loss Scaling
```python
# WRONG: Divide by number of batches
train_loss = total_loss / len(train_loader)
```

**Problem**: Incorrect loss magnitude.

**Solution**: Divide by number of patients (`len(all_patient_preds)`).

### Pitfall 4: Forgetting model.eval()
```python
# WRONG: Model in train mode during validation
for patient_batches in val_loader:
    ...
```

**Problem**: BatchNorm and Dropout behave incorrectly.

**Solution**: Call `model.eval()` before validation.

## Verification Checklist

Before training, verify:

- [ ] Data split at patient level (no overlap)
- [ ] PatientSampler used in DataLoader
- [ ] patient_collate_fn used as collate function
- [ ] MAX (or mean) pooling applied to slice logits
- [ ] Loss computed once per patient
- [ ] Loss scaled by number of patients
- [ ] model.eval() called before validation
- [ ] Metrics computed at patient level

## Summary

Patient-level training is ESSENTIAL for medical imaging:

1. **Split data by patient** → no leakage
2. **Sample patients** → fair batching
3. **Pool slice predictions** → patient-level prediction
4. **Compute loss per patient** → equal contribution
5. **Scale loss correctly** → divide by num_patients

This ensures the model learns to diagnose patients, not just classify individual slices.
