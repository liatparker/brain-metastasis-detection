# Scripts Directory

This directory contains the main Python scripts for training, inference, and data preprocessing.

## Available Scripts

### 1. train.py - Main Training Script

Complete training pipeline with 3-stage curriculum learning.

**Usage:**
```bash
python scripts/train.py \
    --data_root /path/to/brain_metastases/CTs \
    --labels_path /path/to/brain_metastases/labels1.csv \
    --output_path ./outputs \
    --num_epochs 40 \
    --batch_size 32
```

**Arguments:**
- `--data_root`: Root directory containing DICOM files (required)
- `--labels_path`: Path to labels CSV file (required)
- `--output_path`: Directory for saving models and logs (default: ./outputs)
- `--num_epochs`: Number of training epochs (default: 40)
- `--batch_size`: Number of slices per batch (default: 32)
- `--resume`: Path to checkpoint to resume from (optional)
- `--device`: Device to use (default: cuda if available, else cpu)

**What it does:**
1. Loads data and creates patient-level train/val split (80/20)
2. Initializes ResNet50 with RadImageNet weights
3. Trains with automatic 3-stage curriculum:
   - Stage 1 (epochs 0-4): Classifier only
   - Stage 2 (epochs 5-11): Layer4 + Classifier
   - Stage 3 (epochs 12-39): Layer3 + Layer4 + Classifier
4. Saves checkpoints:
   - Best model (when F1 improves)
   - Stage transitions (epochs 5, 12)
   - Final epoch
5. Logs metrics after each epoch

**Output:**
- Models saved to: `{output_path}/models/`
- Best model: `best_model_f1_{score}.pth`
- Stage 1 checkpoint: `checkpoint_epoch05_stage1_f1_{score}.pth`
- Stage 2 checkpoint: `checkpoint_epoch12_stage2_f1_{score}.pth`
- Final checkpoint: `checkpoint_epoch40_stage3_f1_{score}.pth`

**Example**:
```
outputs/models/
├── best_model_f1_0.7234.pth
├── checkpoint_epoch05_stage1_f1_0.6123.pth
├── checkpoint_epoch12_stage2_f1_0.6890.pth
└── checkpoint_epoch40_stage3_f1_0.7150.pth
```

### 2. inference.py - Prediction Script

Run predictions on new CT scans.

**Single Patient:**
```bash
python scripts/inference.py \
    --checkpoint ./outputs/models/best_model_f1_0.7500.pth \
    --input /path/to/patient/dicoms/
```

**Batch Processing:**
```bash
python scripts/inference.py \
    --checkpoint ./outputs/models/best_model_f1_0.7500.pth \
    --input /path/to/all/patients/ \
    --output predictions.csv \
    --batch_process
```

**Arguments:**
- `--checkpoint`: Path to model checkpoint (required)
- `--input`: Path to DICOM directory or file (required)
- `--output`: Output CSV file for predictions (default: predictions.csv)
- `--device`: Device to use (default: cuda if available, else cpu)
- `--batch_process`: Flag to enable batch processing

**Output:**
Single patient prints to console:
```
PREDICTION RESULTS
======================================================================
Prediction: METASTASIS DETECTED
Probability: 0.785
Confidence: HIGH
Number of slices processed: 45
Mean slice probability: 0.623
Max slice probability: 0.785
======================================================================
```

Batch processing saves CSV:
```
patient_id,prediction,probability,num_slices,mean_slice_prob,max_slice_prob
patient_001,1,0.785,45,0.623,0.785
patient_002,0,0.123,38,0.089,0.123
```

### 3. preprocess_data.py - Data Preprocessing Script

Cache preprocessed data for faster training.

**Usage:**
```bash
python scripts/preprocess_data.py \
    --data_root /path/to/brain_metastases/CTs \
    --labels_path /path/to/brain_metastases/labels1.csv \
    --output_path ./preprocessed_data \
    --target_size 256 256
```

**Arguments:**
- `--data_root`: Root directory containing DICOM files (required)
- `--labels_path`: Path to labels CSV file (required)
- `--output_path`: Directory to save preprocessed data (default: ./preprocessed_data)
- `--target_size`: Target image size (default: 256 256)

**What it does:**
1. Loads all DICOM files from labels CSV
2. Applies preprocessing pipeline:
   - Brain extraction
   - Windowing (brain window 40/80 HU)
   - Normalization
   - Resizing
3. Saves preprocessed data as pickle file
4. Creates metadata file with statistics
5. Logs failed slices for debugging

**Output:**
- Preprocessed data: `{output_path}/preprocessed_data.pkl`
- Metadata: `{output_path}/metadata.pkl`
- Failed slices log: `{output_path}/failed_slices.txt`

**Benefits:**
- First training much faster (no repeated preprocessing)
- Consistent preprocessing across runs
- Easy to identify problematic slices

### 4. test_installation.py - Installation Test Script

Verify installation and dependencies.

**Usage:**
```bash
python scripts/test_installation.py
```

**What it tests:**
1. Core dependencies (PyTorch, NumPy, pandas, etc.)
2. Project modules (config, models, data, preprocessing)
3. Configuration system
4. Model creation and forward pass

**Output:**
```
======================================================================
INSTALLATION TEST
======================================================================
Testing imports...
  PyTorch: 2.0.0
  CUDA available: True
  ...
Testing project modules...
  config.Config: OK
  ...
Testing model creation...
  Model created successfully
  Total parameters: 24,583,169
  ...

======================================================================
TEST SUMMARY
======================================================================
Core Dependencies: PASSED
Project Modules: PASSED
Configuration: PASSED
Model Creation: PASSED
======================================================================

All tests PASSED! Installation is correct.
```

## Quick Workflow

### First Time Setup
```bash
# 1. Test installation
python scripts/test_installation.py

# 2. (Optional) Preprocess data
python scripts/preprocess_data.py \
    --data_root /path/to/data \
    --labels_path /path/to/labels.csv

# 3. Train model
python scripts/train.py \
    --data_root /path/to/data \
    --labels_path /path/to/labels.csv \
    --num_epochs 40
```

### Resume Training
```bash
python scripts/train.py \
    --data_root /path/to/data \
    --labels_path /path/to/labels.csv \
    --resume ./outputs/models/checkpoint_epoch12_stage2_*.pth
```

### Make Predictions
```bash
# Single patient
python scripts/inference.py \
    --checkpoint ./outputs/models/best_model_*.pth \
    --input /path/to/patient/

# Batch
python scripts/inference.py \
    --checkpoint ./outputs/models/best_model_*.pth \
    --input /path/to/patients/ \
    --batch_process
```

## Configuration

Modify `config/config.py` to customize hyperparameters instead of using command-line arguments.

See `../config/config.py` for all available options.

## Troubleshooting

### CUDA out of memory
Reduce batch size:
```bash
python scripts/train.py --batch_size 16 ...
```

### No GPU available
Scripts automatically fall back to CPU if CUDA is not available.

### Import errors
Install dependencies:
```bash
pip install -r requirements.txt
```

### Data path errors
Ensure data structure matches expected format:
```
data_root/
├── patient_001/
│   ├── slice_001.dcm
│   └── ...
└── ...
```

## Additional Resources

- Main README: `../README.md`
- Architecture docs: `../docs/ARCHITECTURE.md`
- Preprocessing docs: `../docs/PREPROCESSING.md`
- Quick start guide: `../docs/QUICK_START.md`
- Conversion notes: `../PYTHON_PROJECT_CONVERSION.md`
