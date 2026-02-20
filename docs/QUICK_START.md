# Quick Start Guide

## Installation

```bash
# Clone or navigate to project
cd Deep_Learning_for_Clinical_Neuro_Oncology

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Running the Training Pipeline

### Step 1: Prepare Data

Ensure your data is organized:
```
/path/to/data/brain_metastases/
├── CTs/
│   ├── patient_001/
│   │   ├── slice_001.dcm
│   │   └── ...
│   └── ...
└── labels1.csv
```

### Step 2: (Optional) Preprocess Data

Cache preprocessed data for faster training:
```bash
python scripts/preprocess_data.py \
    --data_root /path/to/brain_metastases/CTs \
    --labels_path /path/to/brain_metastases/labels1.csv \
    --output_path ./preprocessed_data
```

### Step 3: Train Model

Run training:
```bash
python scripts/train.py \
    --data_root /path/to/brain_metastases/CTs \
    --labels_path /path/to/brain_metastases/labels1.csv \
    --output_path ./outputs \
    --num_epochs 40 \
    --batch_size 32
```

Training automatically handles:
- Patient-level data splitting (80/20)
- 3-stage curriculum learning
- Checkpoint saving
- Metric logging

### Step 4: Resume Training (Optional)

Resume from a checkpoint:
```bash
python scripts/train.py \
    --data_root /path/to/brain_metastases/CTs \
    --labels_path /path/to/brain_metastases/labels1.csv \
    --resume ./outputs/models/checkpoint_epoch12_stage2_f1_0.6500.pth
```

## Key Configuration

Modify `config/config.py` to customize hyperparameters:

```python
from config import Config

cfg = Config()

# Data paths
cfg.data.data_root = "/path/to/brain_metastases/CTs"
cfg.data.labels_path = "/path/to/brain_metastases/labels1.csv"
cfg.data.output_path = "./outputs"

# Training
cfg.training.num_epochs = 40
cfg.training.batch_size = 32
cfg.training.classifier_lr = 5e-5
cfg.training.layer4_lr = 1e-5
cfg.training.layer3_lr = 5e-6

# Curriculum stages (automatic)
# Epoch 0-4:   Classifier only
# Epoch 5-11:  Layer4 + Classifier
# Epoch 12-39: Layer3 + Layer4 + Classifier
```

## Expected Training Time

- **Full training** (40 epochs): 3-5 hours on T4 GPU
- **Quick test** (2 epochs): ~10-15 minutes on T4 GPU
- **Stage 1 only** (5 epochs): ~30-45 minutes on T4 GPU

## Checkpoints Saved

Training automatically saves 4-6 checkpoint files at key milestones:

1. **Best Model** (whenever F1 improves):
   - `best_model_f1_<score>.pth`

2. **Stage 1 End** (epoch 5 - classifier-only complete):
   - `checkpoint_epoch05_stage1_f1_<score>.pth`

3. **Stage 2 End** (epoch 12 - Layer4 fine-tuning complete):
   - `checkpoint_epoch12_stage2_f1_<score>.pth`

4. **Final Epoch** (epoch 40 - full training complete):
   - `checkpoint_epoch40_stage3_f1_<score>.pth`

**Example filenames**:
```
best_model_f1_0.7234.pth
checkpoint_epoch05_stage1_f1_0.6123.pth
checkpoint_epoch12_stage2_f1_0.6890.pth
checkpoint_epoch40_stage3_f1_0.7150.pth
```

## Inference

### Single Patient Prediction

```bash
python scripts/inference.py \
    --checkpoint ./outputs/models/best_model_f1_0.7500.pth \
    --input /path/to/patient/dicoms/
```

Output:
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

### Batch Processing

Process multiple patients:
```bash
python scripts/inference.py \
    --checkpoint ./outputs/models/best_model_f1_0.7500.pth \
    --input /path/to/all/patients/ \
    --output predictions.csv \
    --batch_process
```

Output CSV format:
```
patient_id,prediction,probability,num_slices
patient_001,1,0.785,45
patient_002,0,0.123,38
...
```

## Common Issues

### CUDA out of memory
Reduce batch size in config or via command line:
```bash
python scripts/train.py \
    --data_root /path/to/data \
    --labels_path /path/to/labels.csv \
    --batch_size 16
```

### Slow training
Check if GPU is being used:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Specify device explicitly:
```bash
python scripts/train.py \
    --data_root /path/to/data \
    --labels_path /path/to/labels.csv \
    --device cuda
```

### Poor performance
- Verify data paths are correct
- Check class balance in output logs
- Review configuration in `config/config.py`
- Ensure RadImageNet weights are downloading correctly

## Data Format

Expected directory structure:

```
/path/to/data/
├── brain_metastases/
│   ├── CTs/
│   │   ├── patient_001/
│   │   │   ├── slice_001.dcm
│   │   │   ├── slice_002.dcm
│   │   │   └── ...
│   │   ├── patient_002/
│   │   └── ...
│   └── labels1.csv
```

CSV format:
```
ID,Label,PatientID
slice_001,0,patient_001
slice_002,0,patient_001
slice_003,1,patient_002
...
```

Where:
- ID: Slice identifier
- Label: 0 (no metastasis) or 1 (metastasis)
- PatientID: Patient identifier

## Additional Resources

- Full documentation: `README.md`
- Architecture details: `docs/ARCHITECTURE.md`
- Preprocessing pipeline: `docs/PREPROCESSING.md`
- Project structure: `docs/PROJECT_STRUCTURE.md`
- Setup summary: `SETUP_COMPLETE.md`
