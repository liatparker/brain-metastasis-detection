"""
Configuration file for brain metastasis detection training.

This module contains all hyperparameters and paths for the training pipeline.
Modify values here instead of hardcoding in scripts.
"""

import os
from dataclasses import dataclass


@dataclass
class DataConfig:
    """
    Data paths and preprocessing configuration.
    
    Attributes:
        data_root: Root directory containing DICOM files
        labels_path: Path to CSV file with labels
        output_path: Directory for saving models and logs
        target_size: Image size for model input (height, width)
        brain_window_center: CT window center in Hounsfield Units
        brain_window_width: CT window width in Hounsfield Units
    """
    data_root: str = "/path/to/brain_metastases/CTs"
    labels_path: str = "/path/to/brain_metastases/labels1.csv"
    output_path: str = "./outputs"
    
    # Image preprocessing
    target_size: tuple = (256, 256)
    brain_window_center: int = 40
    brain_window_width: int = 80
    
    # Data splitting
    train_ratio: float = 0.8
    random_seed: int = 42


@dataclass
class ModelConfig:
    """
    Model architecture configuration.
    
    Attributes:
        dropout_rate: Dropout probability in classifier
        use_radimagenet: Whether to use RadImageNet pretrained weights
    """
    dropout_rate: float = 0.4
    use_radimagenet: bool = True


@dataclass
class TrainingConfig:
    """
    Training hyperparameters and curriculum learning configuration.
    
    Attributes:
        num_epochs: Total number of training epochs
        batch_size: Number of slices per batch
        num_workers: DataLoader workers (0 for Colab/Windows)
        
        Curriculum Learning:
        stage1_epochs: Epochs for Stage 1 (classifier only)
        stage2_epochs: Epoch to start Stage 2 (unfreeze layer4)
        stage3_epochs: Epoch to start Stage 3 (unfreeze layer3)
        
        Learning Rates:
        classifier_lr: Learning rate for classifier
        layer4_lr: Learning rate for ResNet layer4
        layer3_lr: Learning rate for ResNet layer3
        
        Regularization:
        weight_decay_backbone: L2 regularization for backbone
        weight_decay_classifier: L2 regularization for classifier
        grad_clip_max_norm: Maximum gradient norm for clipping
        
        Loss Configuration:
        pos_weight: Positive class weight in BCE loss
        confidence_weight: Weight for confidence optimization penalty
        target_separation: Target separation between pos/neg predictions
    """
    # Training duration
    num_epochs: int = 40
    batch_size: int = 32
    num_workers: int = 0
    
    # Curriculum learning stages
    stage1_epochs: int = 5   # Classifier only: epochs 0-4
    stage2_epochs: int = 12  # Layer4 unfrozen: epochs 5-11
    stage3_epochs: int = 40  # Layer3 unfrozen: epochs 12-39
    
    # Learning rates (differential)
    classifier_lr: float = 5e-5
    layer4_lr: float = 1e-5
    layer3_lr: float = 5e-6
    
    # Regularization
    weight_decay_backbone: float = 1e-4
    weight_decay_classifier: float = 1e-3
    grad_clip_max_norm: float = 1.0
    
    # Loss function
    pos_weight: float = 1.0
    confidence_weight: float = 0.15
    target_separation: float = 0.30
    
    # Checkpointing
    save_best_only: bool = False  # If True, only save best; if False, save key checkpoints
    checkpoint_dir: str = "checkpoints"


@dataclass
class AugmentationConfig:
    """
    Data augmentation configuration.
    
    Class-conditional augmentation applies different strategies to
    positive (metastasis) and negative (normal) samples.
    
    Attributes:
        aug_prob_positive: Augmentation probability for positive samples
        aug_prob_negative: Augmentation probability for negative samples
        rotation_range: Maximum rotation angle in degrees
        translation_range: Maximum translation in pixels
        gamma_range_positive: Gamma correction range for positives
        gamma_range_negative: Gamma correction range for negatives
        use_clahe: Apply CLAHE to negatives (local contrast enhancement)
        use_sharpening: Apply sharpening filter to negatives
        use_blur: Apply Gaussian blur to positives
    """
    # Augmentation probabilities
    aug_prob_positive: float = 0.7
    aug_prob_negative: float = 0.5
    
    # Geometric transforms
    rotation_range: float = 5.0  # degrees
    translation_range: float = 3.0  # pixels
    
    # Photometric transforms
    gamma_range_positive: tuple = (0.95, 1.05)  # Mild for positives
    gamma_range_negative: tuple = (0.9, 1.1)    # Stronger for negatives
    
    # Texture augmentation
    use_clahe: bool = True  # Negatives only
    use_sharpening: bool = True  # Negatives only
    use_blur: bool = True  # Positives only
    blur_kernel_size: int = 3
    blur_sigma: float = 0.5
    
    # CLAHE parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple = (8, 8)


class Config:
    """
    Main configuration class combining all config sections.
    
    Usage:
        cfg = Config()
        cfg.data.data_root = "/my/data/path"
        cfg.training.num_epochs = 50
    """
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.augmentation = AugmentationConfig()
        
    def update_paths(self, data_root=None, labels_path=None, output_path=None):
        """
        Update data paths.
        
        Args:
            data_root: Root directory for DICOM files
            labels_path: Path to labels CSV
            output_path: Output directory for models and logs
        """
        if data_root:
            self.data.data_root = data_root
        if labels_path:
            self.data.labels_path = labels_path
        if output_path:
            self.data.output_path = output_path
            
        # Create output directories
        os.makedirs(self.data.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.data.output_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.data.output_path, "logs"), exist_ok=True)
    
    def print_config(self):
        """Print current configuration."""
        print("=" * 80)
        print("CONFIGURATION")
        print("=" * 80)
        print("\nData Configuration:")
        for key, value in vars(self.data).items():
            print(f"  {key}: {value}")
        
        print("\nModel Configuration:")
        for key, value in vars(self.model).items():
            print(f"  {key}: {value}")
        
        print("\nTraining Configuration:")
        for key, value in vars(self.training).items():
            print(f"  {key}: {value}")
        
        print("\nAugmentation Configuration:")
        for key, value in vars(self.augmentation).items():
            print(f"  {key}: {value}")
        print("=" * 80)


# Create default configuration instance
default_config = Config()
