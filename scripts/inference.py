"""
Inference script for brain metastasis detection.

This script loads a trained model and makes predictions on new CT scans.

Usage:
    python scripts/inference.py --checkpoint path/to/model.pth --input path/to/dicoms
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pydicom

from config import Config
from src.models.hybrid_model import ResNet50_ImageOnly
from src.preprocessing.advanced_preprocessing import preprocess_ct_slice


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference on CT scans for metastasis detection'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to DICOM directory or single DICOM file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Output CSV file for predictions'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for inference'
    )
    parser.add_argument(
        '--batch_process',
        action='store_true',
        help='Process multiple patients from directory'
    )
    
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Tuple of (model, optimal_threshold)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model
    model = ResNet50_ImageOnly(num_classes=1, dropout_rate=0.4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get optimal threshold
    optimal_threshold = checkpoint.get('optimal_threshold', 0.5)
    
    print(f"Model loaded successfully")
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"Val F1: {checkpoint['val_metrics']['f1']:.4f}")
    
    return model, optimal_threshold


def load_dicom_slice(dicom_path, target_size=(256, 256)):
    """
    Load and preprocess a single DICOM slice.
    
    Args:
        dicom_path: Path to DICOM file
        target_size: Target image size
    
    Returns:
        Preprocessed image tensor of shape [1, H, W]
    """
    try:
        dcm = pydicom.dcmread(dicom_path)
        
        # Preprocess
        preprocessed = preprocess_ct_slice(
            dcm,
            target_size=target_size,
            brain_window_center=40,
            brain_window_width=80
        )
        
        if preprocessed is None:
            return None
        
        # Convert to tensor
        tensor = torch.from_numpy(preprocessed).float()
        
        return tensor
        
    except Exception as e:
        print(f"Error loading {dicom_path}: {e}")
        return None


def predict_patient(model, dicom_paths, device, optimal_threshold):
    """
    Make prediction for a single patient from all their slices.
    
    Args:
        model: Trained model
        dicom_paths: List of paths to DICOM slices for this patient
        device: Device to run inference on
        optimal_threshold: Classification threshold
    
    Returns:
        Dictionary with prediction results
    """
    slice_logits = []
    valid_slices = 0
    
    with torch.no_grad():
        for dicom_path in dicom_paths:
            # Load and preprocess slice
            image = load_dicom_slice(dicom_path)
            
            if image is None:
                continue
            
            # Add batch dimension and move to device
            image = image.unsqueeze(0).to(device)
            
            # Forward pass
            logit = model(image)
            slice_logits.append(logit.squeeze().cpu().item())
            valid_slices += 1
    
    if len(slice_logits) == 0:
        return {
            'prediction': -1,
            'probability': 0.0,
            'patient_logit': 0.0,
            'num_slices': 0,
            'error': 'No valid slices'
        }
    
    # Pool slice logits (MAX pooling)
    patient_logit = max(slice_logits)
    
    # Convert to probability
    patient_prob = 1 / (1 + np.exp(-patient_logit))
    
    # Make prediction
    prediction = 1 if patient_prob > optimal_threshold else 0
    
    return {
        'prediction': prediction,
        'probability': patient_prob,
        'patient_logit': patient_logit,
        'num_slices': valid_slices,
        'mean_slice_prob': np.mean([1/(1+np.exp(-x)) for x in slice_logits]),
        'max_slice_prob': max([1/(1+np.exp(-x)) for x in slice_logits])
    }


def process_single_patient(model, input_path, device, optimal_threshold):
    """
    Process a single patient (directory of DICOM files).
    
    Args:
        model: Trained model
        input_path: Path to directory containing DICOM files
        device: Device to run inference on
        optimal_threshold: Classification threshold
    
    Returns:
        Prediction results dictionary
    """
    if os.path.isdir(input_path):
        # Get all DICOM files
        dicom_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith('.dcm')
        ]
    else:
        # Single file
        dicom_files = [input_path]
    
    print(f"\nProcessing patient with {len(dicom_files)} DICOM files...")
    
    results = predict_patient(model, dicom_files, device, optimal_threshold)
    
    return results


def process_batch(model, input_path, device, optimal_threshold):
    """
    Process multiple patients from directory structure.
    
    Expects structure:
        input_path/
            patient_001/
                slice_001.dcm
                slice_002.dcm
            patient_002/
                ...
    
    Args:
        model: Trained model
        input_path: Root directory containing patient subdirectories
        device: Device to run inference on
        optimal_threshold: Classification threshold
    
    Returns:
        DataFrame with results for all patients
    """
    # Find all patient directories
    patient_dirs = [
        d for d in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, d))
    ]
    
    print(f"\nFound {len(patient_dirs)} patients to process")
    
    results = []
    
    for patient_id in tqdm(patient_dirs, desc="Processing patients"):
        patient_path = os.path.join(input_path, patient_id)
        
        # Get all DICOM files for this patient
        dicom_files = [
            os.path.join(patient_path, f)
            for f in os.listdir(patient_path)
            if f.endswith('.dcm')
        ]
        
        if len(dicom_files) == 0:
            continue
        
        # Make prediction
        pred_results = predict_patient(
            model, dicom_files, device, optimal_threshold
        )
        
        # Add patient ID
        pred_results['patient_id'] = patient_id
        pred_results['num_dicom_files'] = len(dicom_files)
        
        results.append(pred_results)
    
    return pd.DataFrame(results)


def main():
    """Main inference function."""
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    model, optimal_threshold = load_model(args.checkpoint, device)
    
    # Process input
    if args.batch_process:
        # Process multiple patients
        results_df = process_batch(
            model, args.input, device, optimal_threshold
        )
        
        # Save results
        results_df.to_csv(args.output, index=False)
        
        print(f"\nProcessed {len(results_df)} patients")
        print(f"Results saved to: {args.output}")
        
        # Print summary
        print("\nSummary:")
        print(f"Positive predictions: {results_df['prediction'].sum()}")
        print(f"Negative predictions: {(results_df['prediction'] == 0).sum()}")
        print(f"Mean probability: {results_df['probability'].mean():.3f}")
        
    else:
        # Process single patient
        results = process_single_patient(
            model, args.input, device, optimal_threshold
        )
        
        # Print results
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        print(f"Prediction: {'METASTASIS DETECTED' if results['prediction'] == 1 else 'NO METASTASIS'}")
        print(f"Probability: {results['probability']:.3f}")
        print(f"Confidence: {'HIGH' if abs(results['probability'] - 0.5) > 0.3 else 'MODERATE' if abs(results['probability'] - 0.5) > 0.15 else 'LOW'}")
        print(f"Number of slices processed: {results['num_slices']}")
        
        if 'mean_slice_prob' in results:
            print(f"Mean slice probability: {results['mean_slice_prob']:.3f}")
            print(f"Max slice probability: {results['max_slice_prob']:.3f}")
        
        print("="*70)
        
        # Save single result
        results_df = pd.DataFrame([results])
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
