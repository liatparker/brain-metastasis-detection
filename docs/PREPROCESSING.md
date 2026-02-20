# Preprocessing Pipeline

## Overview

The preprocessing pipeline transforms raw DICOM CT scans into normalized, standardized tensors suitable for deep learning. The pipeline includes brain extraction, windowing, quality assessment, and normalization.

## Pipeline Steps

### Step 1: DICOM to Hounsfield Units

Raw DICOM pixel values are converted to Hounsfield Units (HU) using the rescale parameters:

```
HU = pixel_value × RescaleSlope + RescaleIntercept
```

Hounsfield Units provide standardized density measurements:
- Air: -1000 HU
- Water: 0 HU
- Soft tissue: +40 to +80 HU
- Bone: +700 to +3000 HU

### Step 2: Brain Extraction

A binary mask isolates brain tissue from surrounding structures:

**Thresholding:**
```
brain_mask = (HU > -50) & (HU < 100)
```

**Morphological Operations:**
1. Opening: Remove small noise
2. Fill holes: Complete brain region
3. Keep largest connected component

**Purpose:**
- Removes skull, scalp, air
- Focuses analysis on brain parenchyma
- Reduces background noise

### Step 3: Windowing

Brain window optimized for soft tissue and lesion visualization:

**Parameters:**
- Window Center: 40 HU
- Window Width: 80 HU
- Range: 0 to 80 HU

**Formula:**
```
windowed = clip((HU - (center - width/2)) / width, 0, 1)
```

**Effect:**
- Enhances contrast in soft tissue range
- Metastases typically appear as hyperdense (brighter) or hypodense (darker) relative to normal brain
- Reduces influence of extreme values (bone, air)

### Step 4: Normalization

Z-score normalization standardizes intensity distributions:

```
normalized = (windowed - mean) / (std + epsilon)
```

Where:
- mean, std: computed per slice
- epsilon: 1e-8 (prevents division by zero)

**Purpose:**
- Removes scanner-specific variations
- Standardizes intensity ranges across patients
- Improves convergence during training

### Step 5: Resizing

Slices are resized to a fixed dimension:

```
resized = resize(normalized, (256, 256), interpolation=INTER_LINEAR)
```

**Purpose:**
- Consistent input size for neural network
- 256×256 balances resolution and computational cost

### Step 6: Quality Assessment

Each slice is assessed for quality issues:

**Metrics:**
- Brain coverage: Percentage of image containing brain tissue
- Motion artifacts: Blurring or ghosting
- Metal artifacts: Streaking from implants
- Contrast quality: Sufficient dynamic range

**Filters:**
- Insufficient brain: < 5% brain coverage
- Motion artifacts: Edge sharpness below threshold
- Metal artifacts: Streak detection
- Poor contrast: Low standard deviation

**Note:** Asymmetric slices are RETAINED as they may indicate pathology.

## Preprocessing Examples

### Positive Sample (Brain Metastasis)

**Raw DICOM:**
- Hounsfield Units: -1000 to +3000 HU
- Contains skull, air, soft tissue
- Full intensity range visible

**After Brain Extraction:**
- Skull removed via thresholding
- Brain parenchyma isolated
- Background set to zero

**After Windowing:**
- Focused on soft tissue range (0-80 HU)
- Lesion visible as hyperdense region
- Improved contrast around metastasis

**After Normalization:**
- Mean = 0, Std = 1
- Standardized across scanners
- Ready for neural network input

**Key Features:**
- Focal hyperdense lesion (brighter spot)
- Surrounding edema may be visible
- Heterogeneous texture within lesion
- May show irregular margins

### Negative Sample (Normal Brain)

**Raw DICOM:**
- Similar intensity range to positive
- No focal lesions present
- Symmetric brain structure

**After Brain Extraction:**
- Clean brain parenchyma
- No skull or air
- Bilateral symmetry maintained

**After Windowing:**
- Normal gray-white matter differentiation
- No focal abnormalities
- Homogeneous tissue appearance

**After Normalization:**
- Standardized intensities
- Smooth, uniform texture
- Bilateral symmetry preserved

**Key Features:**
- Homogeneous gray matter
- Smooth white matter
- Symmetric hemispheres
- No focal lesions

## Class-Conditional Augmentation

### Positive Samples (Metastases)

**Goal:** Preserve heterogeneous lesion texture while adding robustness

**Augmentations:**
- Geometric: Rotation (±5°), Translation (±3 pixels)
- Mild gamma correction (0.95-1.05): Subtle contrast variation
- Gaussian blur (kernel=3, sigma=0.5): Smooth noise, preserve lesion
- Probability: 70%

**Rationale:**
- Metastases have diagnostic heterogeneous texture
- Aggressive augmentation destroys critical texture patterns
- Mild augmentation maintains clinical relevance

### Negative Samples (Normal Brain)

**Goal:** Make negatives harder by enhancing subtle features

**Augmentations:**
- Geometric: Rotation (±5°), Translation (±3 pixels)
- Strong gamma correction (0.9-1.1): Greater contrast variation
- CLAHE (Contrast Limited AHE): Local contrast enhancement
  - Clip limit: 2.0
  - Tile size: 8×8 pixels
- Sharpening filter: Accentuates edges and boundaries
- Probability: 50%

**Rationale:**
- Normal brain is too uniform and easy to classify
- Enhanced contrast forces model to learn robust features
- Prevents overfitting to smooth, homogeneous appearance
- Simulates scanner variations and subtle artifacts

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

Applied only to negative samples:

**Parameters:**
- Clip limit: 2.0 (prevents over-amplification)
- Tile grid: 8×8 (local regions)

**Effect:**
- Enhances local contrast
- Makes subtle density differences more visible
- Simulates different scanner settings

**Why negatives only:**
- Makes negatives harder to classify
- Prevents model from using "easy" features (uniform appearance)
- Forces learning of more discriminative features

### Sharpening Filter

Applied only to negative samples:

**Kernel:**
```
[[ 0, -1,  0],
 [-1,  5, -1],
 [ 0, -1,  0]]
```

**Effect:**
- Accentuates edges and boundaries
- Emphasizes tissue transitions
- Makes subtle irregularities more visible

**Why negatives only:**
- Metastases have irregular margins (diagnostic feature)
- Enhanced edges on negatives forces model to learn shape-based features
- Prevents reliance solely on texture smoothness


## Quality Control

### Inclusion Criteria
- Brain coverage > 5%
- Adequate contrast (std > threshold)
- No severe motion artifacts
- No severe metal artifacts

### Exclusion Criteria
- Insufficient brain tissue
- Extreme motion blur
- Dense metal artifacts (surgical clips, shunts)

### Special Considerations
**High asymmetry slices are RETAINED:**
- Asymmetry may indicate pathology
- Critical for detecting unilateral lesions
- Excluded from quality filter despite asymmetry

## Data Augmentation Strategy

### Augmentation Probability
- Positive samples: 70% (higher for rare class)
- Negative samples: 50%

### Transform Pipeline (Applied Sequentially)
1. Geometric transforms (rotation, translation)
2. Class-conditional texture augmentation
3. Common photometric adjustments (brightness, contrast)
4. Clipping to valid range

### Augmentation During Training vs Validation
- **Training**: Augmentation active
- **Validation**: No augmentation (evaluate on clean data)

## Implementation Notes

### Memory Efficiency
Preprocessed images are cached in memory during data loading:
- Saves repeated DICOM reading
- Speeds up training significantly
- Stored in `dataset_info` dictionary

### Numerical Stability
- Small epsilon (1e-8) added to denominators
- Clipping applied after each augmentation
- Valid range: [-1, 2] (allows some margin beyond normalized range)

### Threading
- num_workers=0 for DataLoader
- Prevents multiprocessing issues with large cached arrays
- Simpler debugging

## Validation Strategy

### Patient-Level Split
- 80% patients in training
- 20% patients in validation
- No patient appears in both sets
- Stratified by label (maintains class balance)

### No Data Leakage
- PatientID extracted from DICOM metadata
- Splitting done at patient level before slice-level operations
- Ensures generalization to unseen patients

## Preprocessing Output

### Final Tensor Format
```
Input tensor shape: [1, 256, 256]
  - Channel 0: Preprocessed CT slice
  - Data type: float32
  - Value range: Approximately [-3, +3] (z-normalized)
  - Mean: ~0, Std: ~1
```

### Label Format
```
Label tensor shape: [1]
  - 0: No metastasis
  - 1: Metastasis present
  - Data type: float32 (for BCE loss)
```

## Clinical Considerations

### Window Selection
The brain window (40/80 HU) is chosen to:
- Maximize contrast between normal brain and metastases
- Suppress bone and air (not diagnostically relevant)
- Match radiologist viewing settings

### Asymmetry Retention
High asymmetry is a key indicator of pathology:
- Unilateral lesions cause hemispheric asymmetry
- Mass effect shifts midline structures
- Must be preserved in training data

### Quality vs Quantity
While quality filtering removes ~50% of slices:
- Retains all positive samples (rare class)
- Removes only clearly unusable negatives
- Balances data quality with sample size
