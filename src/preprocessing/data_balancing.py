"""
Smart data balancing and quality filtering for imbalanced dataset
Includes patient-level splitting to prevent data leakage
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from tqdm import tqdm
import pickle
import os
import pydicom
from collections import defaultdict


def extract_patient_ids(dataset_info: List[Dict]) -> List[Dict]:
    """
    Extract PatientID from DICOM files to enable patient-level splitting
    
    Args:
        dataset_info: List of dicts with 'path', 'id', 'label'
        
    Returns:
        dataset_info with added 'patient_id' field
    """
    print("Extracting PatientID from DICOM files...")
    
    for sample in tqdm(dataset_info, desc="Reading PatientIDs"):
        try:
            dcm = pydicom.dcmread(sample['path'])
            patient_id = getattr(dcm, 'PatientID', sample['id'])
            sample['patient_id'] = patient_id
        except:
            # Fallback to file ID if DICOM read fails
            sample['patient_id'] = sample['id']
    
    # Analyze patient distribution
    patient_ids = [s['patient_id'] for s in dataset_info]
    unique_patients = len(set(patient_ids))
    
    print(f"✓ Found {unique_patients} unique patients from {len(dataset_info)} samples")
    print(f"  Average samples per patient: {len(dataset_info) / unique_patients:.1f}")
    
    return dataset_info


def split_by_patient(
    dataset_info: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split dataset by patient to prevent data leakage
    
    CRITICAL: Train and validation sets must contain different patients!
    
    Args:
        dataset_info: List with 'patient_id', 'id', 'label'
        train_ratio: Proportion for training (default: 0.8)
        seed: Random seed
        
    Returns:
        train_samples, val_samples (by patient)
    """
    np.random.seed(seed)
    
    # Group samples by patient
    patient_samples = defaultdict(list)
    for sample in dataset_info:
        patient_samples[sample['patient_id']].append(sample)
    
    # Analyze patient-level labels
    patients_with_mets = []
    patients_without_mets = []
    
    for patient_id, samples in patient_samples.items():
        has_positive = any(s['label'] == 1 for s in samples)
        if has_positive:
            patients_with_mets.append(patient_id)
        else:
            patients_without_mets.append(patient_id)
    
    n_patients = len(patient_samples)
    
    print(f"\n{'='*70}")
    print("PATIENT-LEVEL ANALYSIS")
    print(f"{'='*70}")
    print(f"Total unique patients: {n_patients}")
    print(f"  With metastases: {len(patients_with_mets)} ({len(patients_with_mets)/n_patients*100:.1f}%)")
    print(f"  Without metastases: {len(patients_without_mets)}")
    
    # Split patients (not samples!)
    all_patient_ids = list(patient_samples.keys())
    np.random.shuffle(all_patient_ids)
    
    n_train_patients = int(len(all_patient_ids) * train_ratio)
    train_patient_ids = set(all_patient_ids[:n_train_patients])
    val_patient_ids = set(all_patient_ids[n_train_patients:])
    
    # Assign samples based on patient
    train_samples = []
    val_samples = []
    
    for sample in dataset_info:
        if sample['patient_id'] in train_patient_ids:
            train_samples.append(sample)
        else:
            val_samples.append(sample)
    
    # Statistics
    train_pos = sum(1 for s in train_samples if s['label'] == 1)
    val_pos = sum(1 for s in val_samples if s['label'] == 1)
    
    print(f"\n{'='*70}")
    print("PATIENT-LEVEL SPLIT (Prevents Data Leakage)")
    print(f"{'='*70}")
    print(f"\nTrain Set:")
    print(f"  Patients: {len(train_patient_ids)} ({len(train_patient_ids)/n_patients*100:.1f}%)")
    print(f"  Slices: {len(train_samples)}")
    print(f"  Positive slices: {train_pos} ({train_pos/len(train_samples)*100:.2f}%)")
    
    print(f"\nValidation Set:")
    print(f"  Patients: {len(val_patient_ids)} ({len(val_patient_ids)/n_patients*100:.1f}%)")
    print(f"  Slices: {len(val_samples)}")
    print(f"  Positive slices: {val_pos} ({val_pos/len(val_samples)*100:.2f}%)")
    
    # Verify no overlap
    overlap = train_patient_ids & val_patient_ids
    if overlap:
        print(f"\n⚠️ ERROR: {len(overlap)} patients in both train and val!")
        raise ValueError("Patient overlap detected - data leakage!")
    else:
        print(f"\n✓ SUCCESS: Zero patient overlap - data leakage prevented!")
    
    print(f"{'='*70}\n")
    
    return train_samples, val_samples


def filter_negative_samples(
    dataset_info: List[Dict],
    quality_threshold: float = 0.8
) -> Tuple[List[Dict], Dict]:
    """
    Filter negative samples based on quality assessment
    
    Args:
        dataset_info: List of dicts with 'path', 'label', 'quality_report'
        quality_threshold: Fraction of samples to keep (0.8 = keep top 80%)
        
    Returns:
        filtered_data: List of clean samples
        stats: Statistics about filtering
    """
    positive_samples = [d for d in dataset_info if d['label'] == 1]
    negative_samples = [d for d in dataset_info if d['label'] == 0]
    
    print(f"Original dataset: {len(positive_samples)} positive, {len(negative_samples)} negative")
    
    # Keep ALL positives
    filtered_data = positive_samples.copy()
    
    # Filter negatives
    clean_negatives = []
    removed_counts = {
        'insufficient_brain': 0,
        'motion_artifact': 0,
        'metal_artifact': 0,
        'poor_contrast': 0,
        'high_asymmetry': 0,
        'total_removed': 0
    }
    
    for neg_sample in negative_samples:
        quality_report = neg_sample.get('quality_report', {})
        issues = quality_report.get('issues', [])
        
        # IGNORE high_asymmetry - keep those negatives for diversity
        issues_filtered = [i for i in issues if i != 'high_asymmetry']
        
        if len(issues_filtered) == 0:
            clean_negatives.append(neg_sample)
        else:
            for issue in issues:
                if issue in removed_counts:
                    removed_counts[issue] += 1
            removed_counts['total_removed'] += 1
    
    print(f"\nQuality filtering results:")
    for issue, count in removed_counts.items():
        print(f"  {issue}: {count}")
    
    filtered_data.extend(clean_negatives)
    
    stats = {
        'original_positive': len(positive_samples),
        'original_negative': len(negative_samples),
        'filtered_positive': len(positive_samples),
        'filtered_negative': len(clean_negatives),
        'removed': removed_counts
    }
    
    print(f"\nFiltered dataset: {len(positive_samples)} positive, {len(clean_negatives)} negative")
    if len(negative_samples) > 0:
        print(f"Removal rate: {removed_counts['total_removed']/len(negative_samples)*100:.1f}%")
    else:
        print(f"Removal rate: N/A (no negative samples in dataset)")
    
    return filtered_data, stats


def smart_balance_dataset(
    filtered_data: List[Dict],
    positive_augmentation: int = 5,
    target_ratio: float = 3.0,
    seed: int = 42,
    use_patient_split: bool = True,
    no_augmentation: bool = False
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Create balanced training set with smart negative sampling
    
    Strategy:
    1. Split by PATIENT (not slice) to prevent data leakage
    2. Augment positives N× (or skip if no_augmentation=True)
    3. Sample negatives with 50% hard (high asymmetry) + 50% easy (low asymmetry)
    4. Target ratio: 1:N (positive:negative)
    
    Args:
        filtered_data: List of clean samples (must have 'patient_id' field)
        positive_augmentation: Augmentation factor for positives (ignored if no_augmentation=True)
        target_ratio: Negative to positive ratio
        seed: Random seed
        use_patient_split: If True, split by patient (recommended). If False, random split.
        no_augmentation: If True, skip positive augmentation entirely (use with high pos_weight)
        
    Returns:
        train_balanced: Balanced training samples
        val_set: Validation set (kept imbalanced for realistic evaluation)
        stats: Balancing statistics
    """
    np.random.seed(seed)
    
    # STEP 1: Split by patient (prevents data leakage)
    if use_patient_split:
        print("\n🔒 Using PATIENT-LEVEL split (prevents data leakage)")
        train_samples, val_samples = split_by_patient(filtered_data, train_ratio=0.8, seed=seed)
    else:
        print("\n⚠️  Using RANDOM split (may cause data leakage!)")
        # Old random split logic (kept for backward compatibility)
        positives = [d for d in filtered_data if d['label'] == 1]
        negatives = [d for d in filtered_data if d['label'] == 0]
        
        n_pos_val = max(int(len(positives) * 0.2), 1) if len(positives) > 1 else 0
        n_neg_val = max(int(len(negatives) * 0.2), 1) if len(negatives) > 1 else 0
        
        if len(positives) - n_pos_val < 1:
            n_pos_val = max(len(positives) - 1, 0)
        if len(negatives) - n_neg_val < 1:
            n_neg_val = max(len(negatives) - 1, 0)
        
        np.random.shuffle(positives)
        np.random.shuffle(negatives)
        
        train_samples = positives[:-n_pos_val] + negatives[:-n_neg_val] if n_pos_val > 0 else positives + negatives
        val_samples = (positives[-n_pos_val:] if n_pos_val > 0 else []) + (negatives[-n_neg_val:] if n_neg_val > 0 else [])
    
    # STEP 2: Separate train samples by label
    pos_train = [d for d in train_samples if d['label'] == 1]
    neg_train = [d for d in train_samples if d['label'] == 0]
    
    # Handle edge cases
    if len(pos_train) == 0:
        print("WARNING: No positive samples in training set. Cannot train model.")
        return [], val_samples, {'error': 'no_positives'}
    
    if len(neg_train) == 0:
        print("WARNING: No negative samples in training set. Using only positives (not recommended).")
    
    print(f"\nAfter patient split:")
    print(f"  Train: {len(pos_train)} pos, {len(neg_train)} neg")
    print(f"  Val: {len(val_samples)} samples")
    
    # STEP 3: Augment positives (or skip if no_augmentation=True)
    if no_augmentation:
        print(f"\n📊 NO AUGMENTATION MODE: Using ORIGINAL IMBALANCE")
        print(f"   Positives: {len(pos_train)}")
        print(f"   Negatives: {len(neg_train)} (ALL negatives, no sampling)")
        print(f"   Ratio: 1:{len(neg_train)/max(len(pos_train), 1):.1f}")
        print(f"   → Use pos_weight={len(neg_train)/max(len(pos_train), 1):.1f} in loss")
        
        # Use ALL negatives when no augmentation
        train_balanced = pos_train + neg_train
        
        stats = {
            'train_positive': len(pos_train),
            'train_negative': len(neg_train),
            'train_ratio': f"1:{len(neg_train)/len(pos_train):.1f}",
            'val_positive': len([s for s in val_samples if s['label'] == 1]),
            'val_negative': len([s for s in val_samples if s['label'] == 0]),
            'val_ratio': f"1:{len([s for s in val_samples if s['label'] == 0])/max(len([s for s in val_samples if s['label'] == 1]), 1):.1f}",
            'augmentation_factor': 0
        }
        
        return train_balanced, val_samples, stats
    
    # STEP 3B: With augmentation - augment positives and balance negatives
    augmented_positives = []
    for pos in pos_train:
        for aug_idx in range(positive_augmentation):
            aug_sample = pos.copy()
            aug_sample['augmentation_id'] = aug_idx
            augmented_positives.append(aug_sample)
    
    print(f"\nAfter augmentation: {len(augmented_positives)} positive samples")
    
    # Smart negative sampling (only when using augmentation)
    target_negative_count = int(len(augmented_positives) * target_ratio)
    
    # Handle case where we have no negatives or not enough
    if len(neg_train) == 0:
        print(f"\nWARNING: No negative training samples. Training with positives only (not recommended).")
        selected_negatives = []
    elif len(neg_train) < target_negative_count:
        print(f"\nWARNING: Not enough negatives ({len(neg_train)} < {target_negative_count}). Using all available.")
        selected_negatives = neg_train
    else:
        # Sort negatives by asymmetry score (if available)
        negatives_with_scores = []
        for neg in neg_train:
            asymmetry_score = neg.get('quality_report', {}).get('metrics', {}).get('mean_asymmetry', 0)
            negatives_with_scores.append((neg, asymmetry_score))
        
        negatives_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Sample 50% hard (high asymmetry) + 50% easy (low asymmetry)
        n_hard = target_negative_count // 2
        n_easy = target_negative_count - n_hard
        
        hard_negatives = [neg for neg, score in negatives_with_scores[:n_hard]]
        easy_negatives = [neg for neg, score in negatives_with_scores[-n_easy:]]
        
        selected_negatives = hard_negatives + easy_negatives
        
        print(f"\nNegative sampling strategy:")
        print(f"  Hard negatives (high asymmetry): {len(hard_negatives)}")
        print(f"  Easy negatives (low asymmetry): {len(easy_negatives)}")
        print(f"  Total negatives: {len(selected_negatives)}")
    
    # Create balanced training set
    train_balanced = augmented_positives + selected_negatives
    np.random.shuffle(train_balanced)
    
    # Validation set (keep imbalanced for realistic evaluation)
    if use_patient_split:
        # val_samples already defined from split_by_patient
        val_set = val_samples
    else:
        # Use pos_val and neg_val from random split
        val_set = pos_val + neg_val
    np.random.shuffle(val_set)
    
    # Calculate validation stats
    val_pos_count = sum(1 for s in val_set if s['label'] == 1)
    val_neg_count = len(val_set) - val_pos_count
    
    # Calculate ratios safely
    if len(augmented_positives) > 0:
        train_ratio = f"1:{len(selected_negatives)/len(augmented_positives):.1f}"
    else:
        train_ratio = "N/A"
    
    if val_pos_count > 0:
        val_ratio = f"1:{val_neg_count/val_pos_count:.1f}"
    else:
        val_ratio = "N/A"
    
    stats = {
        'train_positive': len(augmented_positives),
        'train_negative': len(selected_negatives),
        'train_ratio': train_ratio,
        'val_positive': val_pos_count,
        'val_negative': val_neg_count,
        'val_ratio': val_ratio,
        'augmentation_factor': positive_augmentation
    }
    
    print(f"\nFinal balanced dataset:")
    print(f"  Train: {len(augmented_positives)} pos : {len(selected_negatives)} neg ({train_ratio})")
    print(f"  Val: {val_pos_count} pos : {val_neg_count} neg (kept imbalanced)")
    
    return train_balanced, val_set, stats


def save_processed_dataset(
    train_data: List[Dict],
    val_data: List[Dict],
    stats: Dict,
    output_dir: str
):
    """
    Save processed dataset info to disk
    
    Args:
        train_data: Training samples info
        val_data: Validation samples info
        stats: Processing statistics
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data lists
    with open(os.path.join(output_dir, 'train_data.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(os.path.join(output_dir, 'val_data.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
    
    # Save statistics
    with open(os.path.join(output_dir, 'processing_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)
    
    # Save readable summary
    with open(os.path.join(output_dir, 'dataset_summary.txt'), 'w') as f:
        f.write("Dataset Processing Summary\n")
        f.write("=" * 50 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nDataset saved to: {output_dir}")
