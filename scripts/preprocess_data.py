"""
Data preprocessing script for brain metastasis detection.

This script preprocesses DICOM files and saves the preprocessed data
for faster training. Run this before training to cache preprocessed data.

Usage:
    python scripts/preprocess_data.py --data_root /path/to/dicoms --labels_path /path/to/labels.csv
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import pydicom

from src.preprocessing.advanced_preprocessing import preprocess_ct_slice


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess CT scan data for training'
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
        default='./preprocessed_data',
        help='Directory to save preprocessed data'
    )
    parser.add_argument(
        '--target_size',
        type=int,
        nargs=2,
        default=[256, 256],
        help='Target image size (height width)'
    )
    
    return parser.parse_args()


def preprocess_dataset(labels_df, dicom_root, target_size, output_path):
    """
    Preprocess all DICOM files and save to disk.
    
    Args:
        labels_df: DataFrame with labels
        dicom_root: Root directory for DICOM files
        target_size: Target image size
        output_path: Where to save preprocessed data
    
    Returns:
        Dictionary mapping slice IDs to preprocessed data
    """
    os.makedirs(output_path, exist_ok=True)
    
    preprocessed_data = {}
    failed_slices = []
    
    print(f"\nPreprocessing {len(labels_df)} slices...")
    
    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        slice_id = row['ID']
        patient_id = row['PatientID']
        label = row['Label']
        
        # Construct DICOM path
        dicom_path = os.path.join(dicom_root, patient_id, f"{slice_id}.dcm")
        
        if not os.path.exists(dicom_path):
            failed_slices.append((slice_id, "File not found"))
            continue
        
        try:
            # Load DICOM
            dcm = pydicom.dcmread(dicom_path)
            
            # Preprocess
            preprocessed = preprocess_ct_slice(
                dcm,
                target_size=tuple(target_size),
                brain_window_center=40,
                brain_window_width=80
            )
            
            if preprocessed is None:
                failed_slices.append((slice_id, "Preprocessing failed"))
                continue
            
            # Store preprocessed data
            preprocessed_data[slice_id] = {
                'image': preprocessed,
                'label': label,
                'patient_id': patient_id
            }
            
        except Exception as e:
            failed_slices.append((slice_id, str(e)))
    
    # Save preprocessed data
    print(f"\nSaving preprocessed data to {output_path}...")
    
    # Save as pickle file
    pickle_path = os.path.join(output_path, 'preprocessed_data.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    # Save metadata
    metadata = {
        'total_slices': len(labels_df),
        'successfully_processed': len(preprocessed_data),
        'failed_slices': len(failed_slices),
        'target_size': target_size
    }
    
    metadata_path = os.path.join(output_path, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save failed slices log
    if failed_slices:
        failed_path = os.path.join(output_path, 'failed_slices.txt')
        with open(failed_path, 'w') as f:
            for slice_id, reason in failed_slices:
                f.write(f"{slice_id}: {reason}\n")
    
    # Print summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"Total slices: {len(labels_df)}")
    print(f"Successfully processed: {len(preprocessed_data)}")
    print(f"Failed: {len(failed_slices)}")
    print(f"Success rate: {100 * len(preprocessed_data) / len(labels_df):.1f}%")
    print(f"Saved to: {pickle_path}")
    print("="*70)
    
    return preprocessed_data


def main():
    """Main preprocessing function."""
    args = parse_args()
    
    print("="*70)
    print("DATA PREPROCESSING")
    print("="*70)
    print(f"Data root: {args.data_root}")
    print(f"Labels path: {args.labels_path}")
    print(f"Output path: {args.output_path}")
    print(f"Target size: {args.target_size}")
    
    # Load labels
    print("\nLoading labels...")
    labels_df = pd.read_csv(args.labels_path)
    print(f"Total samples: {len(labels_df)}")
    
    # Preprocess data
    preprocessed_data = preprocess_dataset(
        labels_df,
        args.data_root,
        args.target_size,
        args.output_path
    )
    
    # Print class distribution
    labels = [v['label'] for v in preprocessed_data.values()]
    print(f"\nClass distribution:")
    print(f"Positive: {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
    print(f"Negative: {len(labels) - sum(labels)} ({100*(len(labels)-sum(labels))/len(labels):.1f}%)")


if __name__ == "__main__":
    main()
