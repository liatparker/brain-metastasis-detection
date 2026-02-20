"""
Advanced visualization utilities for preprocessing and results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple
import os


def visualize_preprocessing_pipeline(
    ct_slice_hu: np.ndarray,
    multi_channel: np.ndarray,
    metadata: Dict,
    label: int,
    save_path: str = None,
    title_prefix: str = ""
):
    """
    Visualize complete preprocessing pipeline
    Shows original → brain mask → windowed → asymmetries
    
    Args:
        ct_slice_hu: Original CT in HU
        multi_channel: [3, H, W] preprocessed channels
        metadata: Dict with intermediate results
        label: Ground truth label
        save_path: Path to save figure
        title_prefix: Prefix for title
    """
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    label_str = "METASTASIS PRESENT" if label == 1 else "NORMAL"
    fig.suptitle(f"{title_prefix}{label_str}", fontsize=16, fontweight='bold')
    
    # Row 1: Original processing
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ct_slice_hu, cmap='gray', vmin=-50, vmax=100)
    ax1.set_title('1. Original CT (HU)', fontsize=12)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(metadata['brain_mask'], cmap='gray')
    ax2.set_title('2. Brain Mask', fontsize=12)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(metadata['windowed'], cmap='gray')
    ax3.set_title('3. Narrow Window [20-50 HU]', fontsize=12)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(metadata['bilateral_filtered'], cmap='gray')
    ax4.set_title('4. Bilateral Filtered', fontsize=12)
    ax4.axis('off')
    
    # Row 2: Asymmetry maps
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(metadata['asymmetry_raw'], cmap='hot', vmin=0, vmax=0.5)
    ax5.set_title(f'5. Raw Asymmetry\n(Mean: {metadata["mean_asymmetry_raw"]:.3f})', fontsize=12)
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(metadata['asymmetry_bilateral'], cmap='hot', vmin=0, vmax=0.5)
    ax6.set_title(f'6. Bilateral Asymmetry\n(Mean: {metadata["mean_asymmetry_bilateral"]:.3f})', fontsize=12)
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    # Asymmetry difference
    ax7 = fig.add_subplot(gs[1, 2])
    asym_diff = metadata['asymmetry_raw'] - metadata['asymmetry_bilateral']
    im7 = ax7.imshow(asym_diff, cmap='seismic', vmin=-0.2, vmax=0.2)
    ax7.set_title('7. Difference\n(Raw - Bilateral)', fontsize=12)
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046)
    
    # Histogram
    ax8 = fig.add_subplot(gs[1, 3])
    brain_mask = metadata['brain_mask']
    ax8.hist(metadata['asymmetry_raw'][brain_mask > 0].flatten(), bins=50, alpha=0.5, label='Raw', color='red')
    ax8.hist(metadata['asymmetry_bilateral'][brain_mask > 0].flatten(), bins=50, alpha=0.5, label='Bilateral', color='blue')
    ax8.set_xlabel('Asymmetry Value')
    ax8.set_ylabel('Frequency')
    ax8.set_title('8. Asymmetry Histogram', fontsize=12)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Row 3: Final channels
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.imshow(multi_channel[0], cmap='gray')
    ax9.set_title('Channel 1:\nNormalized HU', fontsize=12, fontweight='bold')
    ax9.axis('off')
    
    ax10 = fig.add_subplot(gs[2, 1])
    im10 = ax10.imshow(multi_channel[1], cmap='hot')
    ax10.set_title('Channel 2:\nRaw Asymmetry', fontsize=12, fontweight='bold')
    ax10.axis('off')
    plt.colorbar(im10, ax=ax10, fraction=0.046)
    
    ax11 = fig.add_subplot(gs[2, 2])
    im11 = ax11.imshow(multi_channel[2], cmap='hot')
    ax11.set_title('Channel 3:\nBilateral Asymmetry', fontsize=12, fontweight='bold')
    ax11.axis('off')
    plt.colorbar(im11, ax=ax11, fraction=0.046)
    
    # Combined RGB visualization
    ax12 = fig.add_subplot(gs[2, 3])
    # Create RGB image: R=windowed, G=raw_asym, B=bilateral_asym
    rgb_vis = np.stack([
        (multi_channel[0] - multi_channel[0].min()) / (multi_channel[0].max() - multi_channel[0].min() + 1e-8),
        (multi_channel[1] - multi_channel[1].min()) / (multi_channel[1].max() - multi_channel[1].min() + 1e-8),
        (multi_channel[2] - multi_channel[2].min()) / (multi_channel[2].max() - multi_channel[2].min() + 1e-8)
    ], axis=2)
    ax12.imshow(rgb_vis)
    ax12.set_title('Combined Visualization\n(R=HU, G=Raw, B=Bilateral)', fontsize=12, fontweight='bold')
    ax12.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_true_positive_negative_comparison(
    true_pos_samples: List[Tuple],
    true_neg_samples: List[Tuple],
    save_path: str = None
):
    """
    Compare true positives vs true negatives side by side
    
    Args:
        true_pos_samples: List of (ct_hu, multi_channel, metadata) for positives
        true_neg_samples: List of (ct_hu, multi_channel, metadata) for negatives
        save_path: Path to save figure
    """
    n_samples = min(len(true_pos_samples), len(true_neg_samples), 4)
    
    fig, axes = plt.subplots(n_samples, 6, figsize=(24, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # True Positive
        ct_pos, mc_pos, meta_pos = true_pos_samples[i]
        
        axes[i, 0].imshow(ct_pos, cmap='gray', vmin=-50, vmax=100)
        axes[i, 0].set_title(f'TP {i+1}: Original', fontweight='bold')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mc_pos[0], cmap='gray')
        axes[i, 1].set_title('Windowed HU')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(mc_pos[1], cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Raw Asym\n({meta_pos["mean_asymmetry_raw"]:.3f})')
        axes[i, 2].axis('off')
        
        # True Negative
        ct_neg, mc_neg, meta_neg = true_neg_samples[i]
        
        axes[i, 3].imshow(ct_neg, cmap='gray', vmin=-50, vmax=100)
        axes[i, 3].set_title(f'TN {i+1}: Original', fontweight='bold')
        axes[i, 3].axis('off')
        
        axes[i, 4].imshow(mc_neg[0], cmap='gray')
        axes[i, 4].set_title('Windowed HU')
        axes[i, 4].axis('off')
        
        axes[i, 5].imshow(mc_neg[1], cmap='hot', vmin=0, vmax=1)
        axes[i, 5].set_title(f'Raw Asym\n({meta_neg["mean_asymmetry_raw"]:.3f})')
        axes[i, 5].axis('off')
    
    # Add column labels
    fig.text(0.25, 0.98, 'TRUE POSITIVES (Metastasis)', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.75, 0.98, 'TRUE NEGATIVES (Normal)', ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_history(
    history: Dict,
    save_path: str = None
):
    """
    Plot training history: loss, accuracy, sensitivity, specificity
    
    Args:
        history: Dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc',
                'val_sensitivity', 'val_specificity'
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Sensitivity
    axes[1, 0].plot(epochs, history['val_sensitivity'], 'g-', linewidth=2)
    axes[1, 0].axhline(y=0.85, color='orange', linestyle='--', label='Target (85%)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Sensitivity (Recall)')
    axes[1, 0].set_title('Validation Sensitivity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Specificity
    axes[1, 1].plot(epochs, history['val_specificity'], 'm-', linewidth=2)
    axes[1, 1].axhline(y=0.90, color='orange', linestyle='--', label='Target (90%)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Specificity')
    axes[1, 1].set_title('Validation Specificity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix_with_metrics(
    confusion_matrix: np.ndarray,
    metrics: Dict,
    save_path: str = None
):
    """
    Plot confusion matrix with detailed metrics
    
    Args:
        confusion_matrix: 2x2 confusion matrix [[TN, FP], [FN, TP]]
        metrics: Dict with sensitivity, specificity, precision, f1, etc.
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(14, 6))
    
    # Confusion matrix
    ax1 = plt.subplot(1, 2, 1)
    im = ax1.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, int(confusion_matrix[i, j]),
                          ha="center", va="center", color="white" if confusion_matrix[i, j] > confusion_matrix.max()/2 else "black",
                          fontsize=20, fontweight='bold')
    
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Predicted Negative', 'Predicted Positive'])
    ax1.set_yticklabels(['True Negative', 'True Positive'])
    plt.colorbar(im, ax=ax1)
    
    # Metrics table
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')
    
    metrics_text = f"""
    PERFORMANCE METRICS
    {'='*40}
    
    Sensitivity (Recall):  {metrics['sensitivity']:.3f}
    Specificity:           {metrics['specificity']:.3f}
    Precision (PPV):       {metrics['precision']:.3f}
    F1 Score:              {metrics['f1']:.3f}
    
    Accuracy:              {metrics['accuracy']:.3f}
    
    {'='*40}
    
    True Positives (TP):   {int(confusion_matrix[1, 1])}
    True Negatives (TN):   {int(confusion_matrix[0, 0])}
    False Positives (FP):  {int(confusion_matrix[0, 1])}
    False Negatives (FN):  {int(confusion_matrix[1, 0])}
    
    {'='*40}
    
    Clinical Interpretation:
    
    Sensitivity = {metrics['sensitivity']*100:.1f}%
      → Detects {metrics['sensitivity']*100:.1f}% of metastases
    
    Specificity = {metrics['specificity']*100:.1f}%
      → Correctly identifies {metrics['specificity']*100:.1f}% of normals
    """
    
    ax2.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_colored_vs_grayscale(
    channels: np.ndarray,
    save_path: str = None,
    sample_id: str = "Sample"
):
    """
    Visualize colored (human-friendly) vs grayscale (model input) channels
    
    Demonstrates that colors are just for visualization - model sees grayscale numbers
    
    Args:
        channels: [3, H, W] preprocessed channels
        save_path: Path to save figure
        sample_id: Sample identifier for title
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'COLORED (human viewing) vs GRAYSCALE (model input)\n{sample_id}', 
                 fontsize=16, fontweight='bold')
    
    # Row 1: COLORED visualizations (for humans)
    axes[0, 0].imshow(channels[0], cmap='hot', vmin=-1, vmax=2)
    axes[0, 0].set_title('Channel 1: HU\n(COLORED - human friendly)', fontsize=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(channels[1], cmap='hot', vmin=-0.5, vmax=2)
    axes[0, 1].set_title('Channel 2: Raw Asymmetry\n(COLORED - human friendly)', fontsize=10)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(channels[2], cmap='hot', vmin=-0.5, vmax=2)
    axes[0, 2].set_title('Channel 3: Filtered Asymmetry\n(COLORED - human friendly)', fontsize=10)
    axes[0, 2].axis('off')
    
    # Row 2: GRAYSCALE (actual data - what model sees)
    axes[1, 0].imshow(channels[0], cmap='gray', vmin=-1, vmax=2)
    axes[1, 0].set_title('Channel 1: HU\n(GRAYSCALE - model input)', fontsize=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(channels[1], cmap='gray', vmin=-0.5, vmax=2)
    axes[1, 1].set_title('Channel 2: Raw Asymmetry\n(GRAYSCALE - model input)', fontsize=10)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(channels[2], cmap='gray', vmin=-0.5, vmax=2)
    axes[1, 2].set_title('Channel 3: Filtered Asymmetry\n(GRAYSCALE - model input)', fontsize=10)
    axes[1, 2].axis('off')
    
    # Add explanation
    explanation = (
        "KEY POINTS:\n"
        "• TOP ROW: Colored ('hot' colormap) - easier for humans to see patterns\n"
        "• BOTTOM ROW: Grayscale - actual data fed to the model\n"
        "• Both rows show SAME numerical data, just different display\n"
        "• Model only sees numbers (0.0, 0.5, 1.0...), not colors\n"
        "• All 3 channels are grayscale (not RGB color channels)\n"
        f"\nData info: shape={channels.shape}, dtype={channels.dtype}\n"
        f"Memory: {channels.nbytes / 1024:.1f} KB\n"
        f"\nParameter impact (first layer only):\n"
        f"  1 channel:  Conv2d(1, 32, 3×3) = 288 parameters\n"
        f"  3 channels: Conv2d(3, 32, 3×3) = 864 parameters\n"
        f"  Increase: 576 params (0.5% of ~100K total model)"
    )
    fig.text(0.02, 0.02, explanation, fontsize=8, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.15, 1, 0.98])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved colored vs grayscale visualization: {save_path}")
    else:
        plt.show()
    
    plt.close()
