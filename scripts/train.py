"""
Main training script for brain metastasis detection.

This script implements the complete training pipeline including:
- Data loading and preprocessing
- Model initialization with RadImageNet weights
- 3-stage curriculum learning
- Patient-level pooling and loss computation
- Checkpoint saving and evaluation

Usage:
    python scripts/train.py --data_root /path/to/data --labels_path /path/to/labels.csv
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, accuracy_score
import pandas as pd

from config import Config
from src.models.hybrid_model import ResNet50_ImageOnly, load_radimagenet_weights
from src.data.dataset import BrainMetDataset, PatientSampler, patient_collate_fn
from src.preprocessing.data_balancing import create_patient_level_split


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Train brain metastasis detection model'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory containing DICOM files'
    )
    parser.add_argument(
        '--labels_path',
        type=str,
        required=True,
        help='Path to labels CSV file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='./outputs',
        help='Directory for saving models and logs'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=40,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size (number of slices)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--restart_curriculum',
        action='store_true',
        help='When used with --resume, load model weights but restart from epoch 0 (Stage 1)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    
    return parser.parse_args()


def pool_slice_predictions(slice_logits, method='max'):
    """
    Pool slice-level predictions to patient-level.
    
    Args:
        slice_logits: Tensor of shape [num_slices] containing logits
        method: Pooling method ('max' or 'mean')
    
    Returns:
        Single patient-level logit
    """
    if method == 'max':
        return slice_logits.max()
    else:
        return slice_logits.mean()


def compute_confidence_penalty(predictions, labels, target_separation=0.30):
    """
    Compute penalty for insufficient separation between positive and negative predictions.
    
    This encourages the model to make more confident predictions by penalizing
    small separation between mean positive and mean negative probabilities.
    
    Args:
        predictions: List or array of logits
        labels: List or array of binary labels (0 or 1)
        target_separation: Minimum desired separation
    
    Returns:
        Penalty value (0 if separation >= target, positive otherwise)
    """
    if len(predictions) == 0 or len(labels) == 0:
        return 0.0
    
    preds = np.array(predictions)
    labs = np.array(labels)
    
    pos_mask = labs == 1
    neg_mask = labs == 0
    
    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
        # Convert logits to probabilities
        probs = 1 / (1 + np.exp(-preds))
        
        # Calculate separation
        mean_pos = probs[pos_mask].mean()
        mean_neg = probs[neg_mask].mean()
        separation = mean_pos - mean_neg
        
        # Penalty if separation is below target
        if separation < target_separation:
            return (target_separation - separation) ** 2
    
    return 0.0


def compute_metrics(predictions, targets, threshold=0.5):
    """
    Compute classification metrics.
    
    Args:
        predictions: Array of logits
        targets: Array of binary labels
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    probs = 1 / (1 + np.exp(-predictions))
    preds_binary = (probs > threshold).astype(int)
    
    return {
        'f1': f1_score(targets, preds_binary, zero_division=0),
        'sensitivity': recall_score(targets, preds_binary, zero_division=0),
        'specificity': recall_score(targets, preds_binary, pos_label=0, zero_division=0),
        'accuracy': accuracy_score(targets, preds_binary)
    }


def find_optimal_threshold(predictions, targets):
    """
    Find threshold that maximizes F1 score.
    
    Args:
        predictions: Array of logits
        targets: Array of binary labels
    
    Returns:
        Tuple of (optimal_threshold, best_f1)
    """
    thresholds = np.arange(0.05, 0.96, 0.05)
    best_f1 = 0.0
    best_threshold = 0.5
    
    probs = 1 / (1 + np.exp(-predictions))
    
    for thresh in thresholds:
        preds_binary = (probs > thresh).astype(int)
        f1 = f1_score(targets, preds_binary, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold, best_f1


def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """
    Train for one epoch with patient-level pooling.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader with patient-level batching
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        config: Configuration object
    
    Returns:
        Tuple of (loss, predictions, labels)
    """
    model.train()
    total_loss = 0.0
    all_patient_preds = []
    all_patient_labels = []
    
    for patient_batches in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        batch_loss = 0.0
        
        for images, features, label, patient_id in patient_batches:
            images = images.to(device)
            label = label.to(device)
            
            # Forward pass for all slices
            slice_logits = []
            for img in images:
                logit = model(img.unsqueeze(0))
                slice_logits.append(logit.squeeze())
            
            # Pool to patient-level
            patient_logit = pool_slice_predictions(
                torch.stack(slice_logits), 
                method='max'
            )
            
            # Compute loss
            loss = criterion(patient_logit.unsqueeze(0), label)
            batch_loss += loss.item()
            loss.backward()
            
            # Store predictions
            all_patient_preds.append(patient_logit.detach().cpu().item())
            all_patient_labels.append(label.cpu().item())
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=config.training.grad_clip_max_norm
        )
        
        optimizer.step()
        total_loss += batch_loss
    
    # Average loss per patient
    avg_loss = total_loss / len(all_patient_preds)
    
    return avg_loss, all_patient_preds, all_patient_labels


def validate_epoch(model, val_loader, criterion, device):
    """
    Validate for one epoch with patient-level pooling.
    
    Args:
        model: PyTorch model
        val_loader: Validation DataLoader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        Tuple of (loss, predictions, labels)
    """
    model.eval()
    total_loss = 0.0
    val_patient_preds = []
    val_patient_labels = []
    
    with torch.no_grad():
        for patient_batches in tqdm(val_loader, desc="Validation"):
            for images, features, label, patient_id in patient_batches:
                images = images.to(device)
                label = label.to(device)
                
                # Forward pass for all slices
                slice_logits = []
                for img in images:
                    logit = model(img.unsqueeze(0))
                    slice_logits.append(logit.squeeze())
                
                # Pool to patient-level
                patient_logit = pool_slice_predictions(
                    torch.stack(slice_logits), 
                    method='max'
                )
                
                # Compute loss
                loss = criterion(patient_logit.unsqueeze(0), label)
                total_loss += loss.item()
                
                # Store predictions
                val_patient_preds.append(patient_logit.cpu().item())
                val_patient_labels.append(label.cpu().item())
    
    # Average loss per patient
    avg_loss = total_loss / len(val_patient_preds)
    
    return avg_loss, val_patient_preds, val_patient_labels


def update_model_for_stage(model, optimizer, epoch, config):
    """
    Update model freezing and optimizer for curriculum learning stages.
    
    Args:
        model: PyTorch model
        optimizer: Current optimizer
        epoch: Current epoch number
        config: Configuration object
    
    Returns:
        Updated optimizer (or same if no stage change)
    """
    # Stage 2: Unfreeze Layer4
    if epoch == config.training.stage1_epochs:
        print("\n" + "="*70)
        print("STAGE 2: UNFREEZING LAYER4")
        print("="*70)
        
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True
        
        optimizer = optim.Adam([
            {
                'params': model.backbone.layer4.parameters(),
                'lr': config.training.layer4_lr,
                'weight_decay': config.training.weight_decay_backbone
            },
            {
                'params': model.classifier.parameters(),
                'lr': config.training.classifier_lr,
                'weight_decay': config.training.weight_decay_classifier
            }
        ])
        
        print("Layer4 unfrozen")
        print("="*70 + "\n")
    
    # Stage 3: Unfreeze Layer3
    elif epoch == config.training.stage2_epochs:
        print("\n" + "="*70)
        print("STAGE 3: UNFREEZING LAYER3")
        print("="*70)
        
        for param in model.backbone.layer3.parameters():
            param.requires_grad = True
        
        optimizer = optim.Adam([
            {
                'params': model.backbone.layer3.parameters(),
                'lr': config.training.layer3_lr,
                'weight_decay': config.training.weight_decay_backbone
            },
            {
                'params': model.backbone.layer4.parameters(),
                'lr': config.training.layer4_lr,
                'weight_decay': config.training.weight_decay_backbone
            },
            {
                'params': model.classifier.parameters(),
                'lr': config.training.classifier_lr,
                'weight_decay': config.training.weight_decay_classifier
            }
        ])
        
        print("Layer3 unfrozen")
        print("="*70 + "\n")
    
    return optimizer


def save_checkpoint(epoch, model, optimizer, val_metrics, optimal_threshold, 
                   history, save_path, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        epoch: Current epoch number
        model: PyTorch model
        optimizer: Optimizer
        val_metrics: Validation metrics dictionary
        optimal_threshold: Optimal classification threshold
        history: Training history dictionary
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metrics': val_metrics,
        'optimal_threshold': optimal_threshold,
        'history': history
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        print(f"   New best! Saved: {os.path.basename(save_path)}")
    else:
        print(f"   Checkpoint saved: {os.path.basename(save_path)}")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Create configuration
    config = Config()
    config.update_paths(
        data_root=args.data_root,
        labels_path=args.labels_path,
        output_path=args.output_path
    )
    config.training.num_epochs = args.num_epochs
    config.training.batch_size = args.batch_size
    
    # Print configuration
    config.print_config()
    
    # Set device
    device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    
    # Load labels
    print("\nLoading labels...")
    labels_df = pd.read_csv(config.data.labels_path)
    print(f"Total samples: {len(labels_df)}")
    print(f"Positive samples: {labels_df['Label'].sum()}")
    print(f"Negative samples: {(labels_df['Label'] == 0).sum()}")
    
    # Create patient-level split
    print("\nCreating patient-level train/val split...")
    train_df, val_df = create_patient_level_split(
        labels_df,
        train_ratio=config.data.train_ratio,
        random_seed=config.data.random_seed
    )
    print(f"Train patients: {train_df['PatientID'].nunique()}")
    print(f"Val patients: {val_df['PatientID'].nunique()}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = BrainMetDataset(
        labels_df=train_df,
        dicom_root=config.data.data_root,
        target_size=config.data.target_size,
        augment=True,
        config=config
    )
    
    val_dataset = BrainMetDataset(
        labels_df=val_df,
        dicom_root=config.data.data_root,
        target_size=config.data.target_size,
        augment=False,
        config=config
    )
    
    # Create data loaders with patient-level sampling
    print("Creating data loaders...")
    train_sampler = PatientSampler(train_dataset, shuffle=True)
    val_sampler = PatientSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=patient_collate_fn,
        num_workers=config.training.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=patient_collate_fn,
        num_workers=config.training.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print("\nCreating model...")
    model = ResNet50_ImageOnly(
        num_classes=1,
        dropout_rate=config.model.dropout_rate
    )
    
    if config.model.use_radimagenet:
        print("Loading RadImageNet weights...")
        load_radimagenet_weights(model)
    
    # Freeze all backbone layers initially (Stage 1)
    print("Freezing backbone layers (Stage 1: Classifier only)...")
    for i in range(8):
        for param in model.backbone[i].parameters():
            param.requires_grad = False
    
    model = model.to(device)
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([config.training.pos_weight]).to(device)
    )
    
    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=config.training.classifier_lr,
        weight_decay=config.training.weight_decay_classifier
    )
    
    # Initialize training state
    start_epoch = 0
    best_val_f1 = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_sensitivity': [],
        'val_specificity': [],
        'optimal_threshold': [],
        'separation': []
    }
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nLoading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if args.restart_curriculum:
            # Load model weights only, restart curriculum from Stage 1
            print("Restarting curriculum from Stage 1 (epoch 0)")
            print("Model weights loaded, but training will start fresh")
            start_epoch = 0
            best_val_f1 = 0.0
            history = {
                'train_loss': [], 'val_loss': [], 'val_f1': [],
                'val_sensitivity': [], 'val_specificity': [],
                'optimal_threshold': [], 'separation': []
            }
        else:
            # Full resume: load everything and continue from where left off
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_f1 = checkpoint['val_metrics']['f1']
            history = checkpoint['history']
            print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    models_dir = os.path.join(config.data.output_path, "models")
    
    for epoch in range(start_epoch, config.training.num_epochs):
        
        # Update model for curriculum stages
        optimizer = update_model_for_stage(model, optimizer, epoch, config)
        
        # Determine if this is a checkpoint epoch
        save_checkpoint_this_epoch = False
        if epoch == config.training.stage1_epochs - 1:  # Last epoch of stage 1
            save_checkpoint_this_epoch = True
        elif epoch == config.training.stage2_epochs - 1:  # Last epoch of stage 2
            save_checkpoint_this_epoch = True
        elif epoch == config.training.num_epochs - 1:  # Final epoch
            save_checkpoint_this_epoch = True
        
        print("\n" + "="*70)
        print(f"EPOCH {epoch+1}/{config.training.num_epochs}")
        print("="*70)
        
        # Train
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device, config
        )
        
        # Compute confidence penalty
        conf_penalty = compute_confidence_penalty(
            train_preds,
            train_labels,
            target_separation=config.training.target_separation
        ) if config.training.confidence_weight > 0 else 0.0
        
        # Validate
        val_loss, val_preds, val_labels = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Convert to arrays
        val_preds_arr = np.array(val_preds)
        val_labels_arr = np.array(val_labels)
        
        # Find optimal threshold
        optimal_threshold, _ = find_optimal_threshold(val_preds_arr, val_labels_arr)
        
        # Compute metrics
        val_metrics = compute_metrics(val_preds_arr, val_labels_arr, optimal_threshold)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_metrics['f1'])
        history['val_sensitivity'].append(val_metrics['sensitivity'])
        history['val_specificity'].append(val_metrics['specificity'])
        history['optimal_threshold'].append(optimal_threshold)
        
        # Print results
        print(f"\nEpoch {epoch+1} Results (threshold={optimal_threshold:.2f}):")
        print(f"   Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
        print(f"   Val F1: {val_metrics['f1']:.4f}  " +
              f"Sens: {val_metrics['sensitivity']:.4f}  " +
              f"Spec: {val_metrics['specificity']:.4f}")
        
        # Compute and print separation
        val_probs = 1 / (1 + np.exp(-val_preds_arr))
        pos_mask = val_labels_arr == 1
        neg_mask = val_labels_arr == 0
        
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            mean_pos = val_probs[pos_mask].mean()
            mean_neg = val_probs[neg_mask].mean()
            separation = mean_pos - mean_neg
            
            print(f"   Mean pred POS: {mean_pos:.3f}  " +
                  f"NEG: {mean_neg:.3f}  " +
                  f"Sep: {separation:.3f}")
            
            history['separation'].append(separation)
            
            if config.training.confidence_weight > 0:
                print(f"   Confidence penalty: {conf_penalty:.4f}")
        
        # Save checkpoints
        # Always save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            save_path = os.path.join(
                models_dir,
                f"best_model_f1_{best_val_f1:.4f}.pth"
            )
            save_checkpoint(
                epoch, model, optimizer, val_metrics,
                optimal_threshold, history, save_path, is_best=True
            )
        
        # Save at curriculum switches and final epoch
        if save_checkpoint_this_epoch:
            stage_name = (
                "stage1" if epoch < config.training.stage1_epochs
                else "stage2" if epoch < config.training.stage2_epochs
                else "stage3"
            )
            save_path = os.path.join(
                models_dir,
                f"checkpoint_epoch{epoch+1:02d}_{stage_name}_f1_{val_metrics['f1']:.4f}.pth"
            )
            save_checkpoint(
                epoch, model, optimizer, val_metrics,
                optimal_threshold, history, save_path, is_best=False
            )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best Val F1: {best_val_f1:.4f}")
    print(f"Models saved to: {models_dir}")


if __name__ == "__main__":
    main()
