# Project Structure

## Directory Organization

```
Deep_Learning_for_Clinical_Neuro_Oncology/
│
├── scripts/
│   ├── train.py
│   │   Main training script with curriculum learning
│   ├── inference.py
│   │   Inference on new CT scans
│   ├── preprocess_data.py
│   │   Cache preprocessed data
│   └── test_installation.py
│       Verify installation
│
├── config/
│   ├── config.py
│   │   Configuration classes:
│   │   - DataConfig
│   │   - ModelConfig
│   │   - TrainingConfig
│   │   - AugmentationConfig
│   └── __init__.py
│
├── src/
│   ├── preprocessing/
│   │   ├── advanced_preprocessing.py
│   │   │   Core preprocessing functions:
│   │   │   - extract_brain_mask(): Brain tissue isolation
│   │   │   - apply_window(): HU windowing
│   │   │   - assess_quality(): Quality filtering
│   │   │   - preprocess_ct_slice(): Complete pipeline
│   │   │
│   │   └── data_balancing.py
│   │       Patient-level train/val splitting
│   │
│   ├── models/
│   │   └── hybrid_model.py
│   │       Model definitions:
│   │       - ResNet50_ImageOnly: Main image-only model
│   │       - load_radimagenet_weights(): Load medical pretrained weights
│   │
│   ├── data/
│   │   ├── dataset.py
│   │   │   PyTorch Dataset classes:
│   │   │   - BrainMetDataset: Image-only dataset
│   │   │   - PatientSampler: Patient-level sampling
│   │   │   - patient_collate_fn: Patient grouping
│   │   │
│   │   └── dicom_loader.py
│   │       DICOM reading and parsing
│   │
│   └── utils/
│       └── visualization_advanced.py
│           Metrics and visualization:
│           - compute_metrics()
│           - plot_training_history()
│           - plot_confusion_matrix()
│
├── docs/
│   ├── ARCHITECTURE.md
│   │   Model architecture and training strategy
│   │
│   ├── PREPROCESSING.md
│   │   Preprocessing and augmentation pipeline
│   │
│   ├── PATIENT_LEVEL_TRAINING.md
│   │   Patient-level methodology
│   │
│   ├── PROJECT_STRUCTURE.md
│   │   This file
│   │
│   └── QUICK_START.md
│       Quick start guide
│
├── examples/
│   └── generate_examples.py
│       Generate preprocessing visualization
│
├── setup.py
│   Package installation
│
├── .gitignore
│   Git ignore patterns
│
├── requirements.txt
│   Python dependencies
│
└── README.md
    Main documentation
```

## Code Organization Principles

### 1. Separation of Concerns
- **Preprocessing**: All data preparation in `src/preprocessing/`
- **Models**: Architecture definitions in `src/models/`
- **Training**: Command-line scripts orchestrate full pipeline
- **Data**: Dataset and sampling logic in `src/data/`
- **Configuration**: Centralized in `config/`
- **Utilities**: Reusable functions in `src/utils/`

### 2. Modularity
Each module has a single, clear responsibility:
- `advanced_preprocessing.py`: Image preprocessing pipeline
- `data_balancing.py`: Patient-level data splitting
- `hybrid_model.py`: Model architecture (ResNet50_ImageOnly)
- `dataset.py`: Dataset classes and patient-level sampling
- `train.py`: Training orchestration
- `inference.py`: Prediction pipeline

### 3. Reusability
Functions are designed for reuse:
- Can be imported into notebooks or scripts
- Self-contained with clear inputs/outputs
- Minimal dependencies between modules

## Key Files

### Training Script
`scripts/train.py`

**Purpose**: Main training pipeline with command-line interface

**What it does**:
1. Parse command-line arguments
2. Load and validate data
3. Create patient-level train/val split
4. Initialize model with RadImageNet weights
5. Run 3-stage curriculum learning
6. Save checkpoints at key milestones
7. Log metrics and training progress

**Usage**: 
```bash
python scripts/train.py --data_root /path --labels_path /path --num_epochs 40
```

### Model Definition
`src/models/hybrid_model.py`

**Purpose**: Model architecture definitions

**Key Classes**:
- `ResNet50_ImageOnly`: Main classification model
  - Single-channel CT input [B, 1, 256, 256]
  - ResNet-50 backbone (RadImageNet pretrained)
  - Classifier: Linear(2048 → 512) → ReLU → Dropout → Linear(512 → 1)

**Key Functions**:
- `load_radimagenet_weights(model)`: Load medical pretrained weights from HuggingFace

### Preprocessing
`src/preprocessing/advanced_preprocessing.py`

**Purpose**: DICOM preprocessing pipeline

**Key Functions**:
- `extract_brain_mask(ct_hu)`: Binary brain mask extraction
- `apply_window(ct_hu, level, width)`: HU windowing
- `preprocess_ct_slice(dcm, target_size, ...)`: Complete preprocessing pipeline
- `assess_quality(ct_hu)`: Quality assessment and filtering

`src/preprocessing/data_balancing.py`

**Purpose**: Patient-level data splitting

**Key Functions**:
- `create_patient_level_split(labels_df, train_ratio, ...)`: Split at patient level (no leakage)

### Dataset
`src/data/dataset.py`

**Key Classes**:
- `BrainMetDataset`: Image-only dataset for patient-level training
- `PatientSampler`: Patient-level batch sampling
- `patient_collate_fn`: Groups slices by patient

## Configuration Management

### Hyperparameters
Defined in notebook cells:
- Learning rates (differential per layer)
- Batch size
- Number of epochs
- Augmentation parameters
- Loss weights

### Paths
Set in notebook setup:
- `DATA_PATH`: DICOM directory
- `OUTPUT_PATH`: Model checkpoints and logs
- `DRIVE_PATH`: Google Drive mount (if using Colab)

### Model Weights
RadImageNet weights loaded from HuggingFace:
- Automatically downloaded on first use
- Cached for subsequent runs

## Data Flow

### Training Pipeline

```
DICOM files
    ↓
Load and convert to HU
    ↓
Brain extraction
    ↓
Windowing
    ↓
Quality assessment
    ↓
Train/Val split (patient-level)
    ↓
PyTorch Dataset + DataLoader
    ↓
Model (ResNet50-ImageOnly)
    ↓
Training loop (curriculum)
    ↓
Validation
    ↓
Checkpoint saving
    ↓
Evaluation and visualization
```

### Inference Pipeline

```
New DICOM
    ↓
Preprocess (same as training)
    ↓
Load best checkpoint
    ↓
Forward pass (all slices)
    ↓
MAX pooling
    ↓
Threshold (optimal)
    ↓
Patient prediction
```

## Testing and Validation

### Quality Checks
- `test_pos_neg_samples.py`: Verify positive/negative loading
- `quick_test_samples.py`: Quick sanity check

### Model Verification
Run these checks before training:
1. Model architecture: `print(model)`
2. Trainable parameters: `sum(p.numel() for p in model.parameters() if p.requires_grad)`
3. Sample batch: Process one batch and check shapes

### Data Verification
Run these checks before training:
1. Class balance: Count positives and negatives
2. Patient split: Verify no overlap between train/val
3. Augmentation: Visualize augmented samples

## Extension Points

### Adding New Augmentations
Edit `src/data/dataset.py`, `BrainMetHybridDataset.__getitem__()`:
- Add augmentation in class-conditional block
- Test on both positive and negative samples

### Adding New Augmentations
Edit dataset class in `src/data/dataset.py`:
- Add augmentation logic to `BrainMetDataset.__getitem__`
- Update `AugmentationConfig` in `config/config.py`
- Test on both positive and negative samples
- Consider class-conditional strategy (different for pos/neg)

### Adding New Models
1. Define in `src/models/hybrid_model.py`
2. Update `create_model()` factory function
3. Test with sample batch

### Custom Loss Functions
Define in notebook training loop:
- Implement loss computation
- Integrate into training loop
- Log custom metrics

## Dependencies

### Core
- PyTorch: Deep learning framework
- torchvision: ResNet architecture
- numpy: Numerical operations

### Medical Imaging
- pydicom: DICOM reading
- SimpleITK: Medical image processing
- nibabel: Neuroimaging formats

### Image Processing
- opencv-python: Image operations
- scikit-image: Advanced processing
- Pillow: Image I/O

### Utilities
- pandas: Data manipulation
- matplotlib/seaborn: Visualization
- scikit-learn: Metrics

See `requirements.txt` for full list with versions.

## Performance Optimization

### Memory
- Use `num_workers=0` for DataLoader (Colab constraint)
- Cache preprocessed images in RAM
- Delete large objects after use

### Speed
- Use GPU for training (Colab T4 or better)
- Batch processing of slices
- Efficient augmentation (avoid redundant operations)

### Disk
- Store checkpoints on Google Drive (if using Colab)
- Compress large files
- Clean up temporary files

## Troubleshooting

### Common Issues

**CUDA out of memory:**
- Reduce batch size
- Use gradient accumulation
- Clear cache: `torch.cuda.empty_cache()`

**Slow training:**
- Check GPU utilization
- Reduce augmentation complexity
- Increase num_workers (if not on Colab)

**Poor performance:**
- Check class balance
- Verify preprocessing
- Review augmentation strategy
- Check learning rates

**Model collapse:**
- Reduce learning rates
- Freeze more layers
- Check gradient flow
- Verify loss scaling

See `docs/ARCHITECTURE.md` for detailed troubleshooting guide.
