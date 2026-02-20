"""
Advanced preprocessing for brain metastasis detection
3-channel approach: Normalized HU + Raw Asymmetry + Bilateral Asymmetry
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, label, binary_fill_holes, binary_erosion, binary_dilation
from typing import Tuple, Dict, List


def extract_brain_mask(ct_slice_hu: np.ndarray) -> np.ndarray:
    """
    Fast brain extraction without external models
    
    Args:
        ct_slice_hu: CT slice in Hounsfield Units [H, W]
        
    Returns:
        brain_mask: Binary mask [H, W]
    """
    # Threshold brain tissue (0-100 HU)
    brain_mask = (ct_slice_hu > 0) & (ct_slice_hu < 100)
    
    # Morphological operations
    brain_mask = binary_fill_holes(brain_mask)
    brain_mask = binary_erosion(brain_mask, iterations=2)
    brain_mask = binary_dilation(brain_mask, iterations=2)
    
    # Keep largest component (brain)
    components, n_components = label(brain_mask)
    if n_components > 0:
        component_sizes = [(i, (components == i).sum()) for i in range(1, n_components + 1)]
        largest_component = max(component_sizes, key=lambda x: x[1])[0]
        brain_mask = (components == largest_component)
    
    return brain_mask.astype(np.uint8)


def apply_window(hu_image: np.ndarray, level: float, width: float) -> np.ndarray:
    """
    Apply HU windowing
    
    Args:
        hu_image: Image in Hounsfield Units
        level: Window center (L)
        width: Window width (W)
        
    Returns:
        windowed: Windowed image [0, 1]
    """
    img_min = level - width / 2
    img_max = level + width / 2
    
    windowed = np.clip(hu_image, img_min, img_max)
    windowed = (windowed - img_min) / (img_max - img_min)
    
    return windowed


def compute_asymmetry_map_fast(image: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """
    Fast asymmetry computation between left and right hemispheres
    
    Args:
        image: Preprocessed image (windowed, normalized)
        brain_mask: Binary brain mask
        
    Returns:
        asymmetry_map: Absolute difference map
    """
    # Find midline
    y_coords, x_coords = np.where(brain_mask)
    if len(x_coords) == 0:
        return np.zeros_like(image)
    
    midline_x = int(x_coords.mean())
    
    # Split hemispheres
    left = image[:, :midline_x]
    right = image[:, midline_x:]
    right_mirrored = np.fliplr(right)
    
    # Match width
    min_width = min(left.shape[1], right_mirrored.shape[1])
    left_crop = left[:, :min_width]
    right_crop = right_mirrored[:, :min_width]
    
    # Compute asymmetry
    asymmetry = np.abs(left_crop - right_crop)
    
    # Reconstruct full image - place symmetrically around midline
    asymmetry_full = np.zeros_like(image)
    
    # Place asymmetry on both sides of midline
    # Ensure we don't exceed image boundaries
    left_start = max(0, midline_x - min_width)
    left_end = midline_x
    right_start = midline_x
    right_end = min(image.shape[1], midline_x + min_width)
    
    # Adjust asymmetry if needed for boundary cases
    actual_left_width = left_end - left_start
    actual_right_width = right_end - right_start
    
    if actual_left_width > 0:
        asymmetry_full[:, left_start:left_end] = asymmetry[:, :actual_left_width]
    if actual_right_width > 0:
        asymmetry_full[:, right_start:right_end] = np.fliplr(asymmetry[:, :actual_right_width])
    
    asymmetry_full *= brain_mask
    
    return asymmetry_full


def preprocess_slice_3channel(
    ct_input,
    brain_mask: np.ndarray = None,
    window_level: float = 35.0,
    window_width: float = 30.0
) -> Tuple[np.ndarray, Dict]:
    """
    Create 3-channel preprocessing for subtle metastasis detection
    
    Channel 1: Normalized narrow-windowed HU
    Channel 2: Raw asymmetry map
    Channel 3: Bilateral-filtered asymmetry map
    
    Args:
        ct_input: Either DICOM object or CT slice in Hounsfield Units [H, W]
        brain_mask: Optional pre-computed brain mask
        window_level: Center of HU window (default: 35 for parenchyma)
        window_width: Width of HU window (default: 30 for narrow window)
        
    Returns:
        multi_channel: [3, H, W] preprocessed channels
        metadata: Dict with intermediate results for visualization
    """
    # Convert DICOM to HU if needed
    if hasattr(ct_input, 'pixel_array'):
        # It's a DICOM object
        pixels = ct_input.pixel_array.astype(np.float32)
        slope = getattr(ct_input, 'RescaleSlope', 1.0)
        intercept = getattr(ct_input, 'RescaleIntercept', 0.0)
        ct_slice_hu = pixels * slope + intercept
    else:
        # It's already a numpy array
        ct_slice_hu = ct_input
    
    # Extract brain mask if not provided
    if brain_mask is None:
        brain_mask = extract_brain_mask(ct_slice_hu)
    
    # Apply brain mask
    brain_hu = ct_slice_hu * brain_mask
    
    # ===== CHANNEL 1: Narrow window (parenchymal detail) =====
    windowed = apply_window(brain_hu, level=window_level, width=window_width)
    windowed_norm = (windowed - windowed.mean()) / (windowed.std() + 1e-8)
    channel_1 = windowed_norm
    
    # ===== CHANNEL 2: Raw asymmetry (maximum sensitivity) =====
    asymmetry_raw = compute_asymmetry_map_fast(windowed_norm, brain_mask)
    asymmetry_raw_norm = (asymmetry_raw - asymmetry_raw.mean()) / (asymmetry_raw.std() + 1e-8)
    channel_2 = asymmetry_raw_norm
    
    # ===== CHANNEL 3: Bilateral-filtered asymmetry (noise-robust) =====
    windowed_bilateral = cv2.bilateralFilter(
        windowed_norm.astype(np.float32),
        d=5,              # Fast mode
        sigmaColor=25,    # Noise removal
        sigmaSpace=1.5    # Preserve small structures (3-5mm metastases)
    )
    asymmetry_bilateral = compute_asymmetry_map_fast(windowed_bilateral, brain_mask)
    asymmetry_bilateral_norm = (asymmetry_bilateral - asymmetry_bilateral.mean()) / (asymmetry_bilateral.std() + 1e-8)
    channel_3 = asymmetry_bilateral_norm
    
    # Stack channels
    multi_channel = np.stack([channel_1, channel_2, channel_3], axis=0)
    
    # Metadata for visualization
    metadata = {
        'brain_mask': brain_mask,
        'windowed': windowed,
        'windowed_norm': windowed_norm,
        'bilateral_filtered': windowed_bilateral,
        'asymmetry_raw': asymmetry_raw,
        'asymmetry_bilateral': asymmetry_bilateral,
        'mean_asymmetry_raw': asymmetry_raw[brain_mask > 0].mean() if brain_mask.sum() > 0 else 0,
        'mean_asymmetry_bilateral': asymmetry_bilateral[brain_mask > 0].mean() if brain_mask.sum() > 0 else 0
    }
    
    return multi_channel, metadata


def assess_slice_quality(
    ct_slice_hu: np.ndarray,
    brain_mask: np.ndarray,
    label: int = 0
) -> Dict:
    """
    Assess quality of a CT slice
    Identify artifacts, poor contrast, and other issues
    
    Args:
        ct_slice_hu: CT slice in Hounsfield Units
        brain_mask: Binary brain mask
        label: Ground truth label (0 or 1)
        
    Returns:
        quality_report: Dict with quality metrics and issues
    """
    quality_issues = []
    metrics = {}
    
    brain_hu = ct_slice_hu[brain_mask > 0]
    
    if len(brain_hu) == 0:
        return {
            'is_good': False,
            'issues': ['no_brain_tissue'],
            'metrics': {}
        }
    
    # Check 1: Brain mask coverage (too little brain tissue)
    brain_coverage = brain_mask.sum() / (brain_mask.shape[0] * brain_mask.shape[1])
    metrics['brain_coverage'] = brain_coverage
    if brain_coverage < 0.08:  # Relaxed: Only flag truly empty slices
        quality_issues.append('insufficient_brain')
    
    # Check 2: Severe motion artifacts (extremely high local variance)
    patch_size = 10
    local_stds = []
    for i in range(0, ct_slice_hu.shape[0] - patch_size, patch_size):
        for j in range(0, ct_slice_hu.shape[1] - patch_size, patch_size):
            patch = ct_slice_hu[i:i+patch_size, j:j+patch_size]
            if brain_mask[i:i+patch_size, j:j+patch_size].sum() > 50:
                local_stds.append(np.std(patch))
    
    if len(local_stds) > 0:
        local_std_variation = np.std(local_stds)
        metrics['local_std_variation'] = local_std_variation
        # Realistic threshold: Normal brain CTs have variation 30-80, flag only severe motion
        if local_std_variation > 200:
            quality_issues.append('motion_artifact')
    
    # Check 3: Metal artifacts (extreme HU values)
    extreme_count = ((brain_hu > 500) | (brain_hu < -200)).sum()
    metrics['extreme_hu_count'] = extreme_count
    if extreme_count > 100:
        quality_issues.append('metal_artifact')
    
    # Check 4: Poor contrast (very flat image)
    hu_range = brain_hu.max() - brain_hu.min()
    metrics['hu_range'] = hu_range
    if hu_range < 25:  # Very poor contrast only
        quality_issues.append('poor_contrast')
    
    # Check 5: Extreme asymmetry (for negatives only - may indicate unlabeled pathology)
    if label == 0:
        windowed = apply_window(ct_slice_hu * brain_mask, level=35, width=30)
        asymmetry = compute_asymmetry_map_fast(windowed, brain_mask)
        mean_asymmetry = asymmetry[brain_mask > 0].mean() if brain_mask.sum() > 0 else 0
        metrics['mean_asymmetry'] = mean_asymmetry
        
        # Flag only very extreme asymmetry that suggests unlabeled lesion
        # Threshold 0.35: Allows natural anatomical variation while catching obvious pathology
        if mean_asymmetry > 0.35:
            quality_issues.append('high_asymmetry')
    
    # Overall quality assessment
    is_good = len(quality_issues) == 0
    
    return {
        'is_good': is_good,
        'issues': quality_issues,
        'metrics': metrics
    }


def create_2_5d_stack(
    slices: List[np.ndarray],
    center_idx: int,
    n_slices: int = 5
) -> np.ndarray:
    """
    Create 2.5D stack centered at given index
    
    Args:
        slices: List of preprocessed slices [3, H, W]
        center_idx: Index of center slice
        n_slices: Number of slices in stack (default: 5)
        
    Returns:
        stack: [n_slices, 3, H, W]
    """
    half_slices = n_slices // 2
    stack = []
    
    for offset in range(-half_slices, half_slices + 1):
        idx = center_idx + offset
        if 0 <= idx < len(slices):
            stack.append(slices[idx])
        else:
            # Pad with zeros if out of bounds
            stack.append(np.zeros_like(slices[0]))
    
    return np.stack(stack, axis=0)
