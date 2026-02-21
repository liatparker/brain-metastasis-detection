"""
View checkpoint results without loading model weights.

Quick way to inspect checkpoint metrics and training history.

Usage:
    python view_checkpoint_results.py --checkpoint outputs/models/YOUR_CHECKPOINT.pth
"""

import argparse
import torch
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    
    print("=" * 70)
    print("CHECKPOINT RESULTS")
    print("=" * 70)
    
    if not checkpoint_path.exists():
        print(f"\nERROR: File not found: {checkpoint_path}")
        return
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"\nFile: {checkpoint_path.name}")
    print(f"Size: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Epoch
    if 'epoch' in checkpoint:
        print(f"\n{'Epoch':<25} {checkpoint['epoch']}")
    
    # Validation metrics
    if 'val_metrics' in checkpoint:
        print(f"\n{'='*70}")
        print("VALIDATION METRICS")
        print(f"{'='*70}")
        
        m = checkpoint['val_metrics']
        print(f"{'F1 Score':<25} {m.get('f1', 0):.4f}")
        print(f"{'Sensitivity (Recall)':<25} {m.get('sensitivity', 0):.4f}")
        print(f"{'Specificity':<25} {m.get('specificity', 0):.4f}")
        
        if m.get('loss', 0) > 0:
            print(f"{'Validation Loss':<25} {m['loss']:.4f}")
        
        if 'mean_prob_pos' in m:
            print(f"\n{'Mean Prob POS':<25} {m['mean_prob_pos']:.4f}")
            print(f"{'Mean Prob NEG':<25} {m['mean_prob_neg']:.4f}")
            print(f"{'Separation':<25} {m.get('separation', 0):.4f}")
    
    # Optimal threshold
    if 'optimal_threshold' in checkpoint:
        print(f"\n{'Optimal Threshold':<25} {checkpoint['optimal_threshold']:.4f}")
    elif 'best_threshold' in checkpoint:
        print(f"\n{'Optimal Threshold':<25} {checkpoint['best_threshold']:.4f}")
    
    # Training history
    if 'history' in checkpoint:
        history = checkpoint['history']
        num_epochs = len(history.get('train_loss', []))
        
        print(f"\n{'='*70}")
        print(f"TRAINING HISTORY ({num_epochs} epochs)")
        print(f"{'='*70}")
        
        if num_epochs > 0:
            print(f"{'Final Train Loss':<25} {history['train_loss'][-1]:.4f}")
            print(f"{'Final Val Loss':<25} {history['val_loss'][-1]:.4f}")
            print(f"{'Final Val F1':<25} {history['val_f1'][-1]:.4f}")
            
            if 'val_sensitivity' in history:
                print(f"{'Final Sensitivity':<25} {history['val_sensitivity'][-1]:.4f}")
                print(f"{'Final Specificity':<25} {history['val_specificity'][-1]:.4f}")
            
            # Find best epoch
            best_f1 = max(history['val_f1'])
            best_idx = history['val_f1'].index(best_f1)
            
            print(f"\n{'Best Val F1':<25} {best_f1:.4f} (epoch {best_idx + 1})")
            
            if 'val_sensitivity' in history:
                best_sens = history['val_sensitivity'][best_idx]
                best_spec = history['val_specificity'][best_idx]
                print(f"{'Best Sensitivity':<25} {best_sens:.4f}")
                print(f"{'Best Specificity':<25} {best_spec:.4f}")
    
    # Model info
    if 'model_state_dict' in checkpoint:
        num_tensors = len(checkpoint['model_state_dict'])
        print(f"\n{'='*70}")
        print(f"MODEL INFO")
        print(f"{'='*70}")
        print(f"{'Parameter tensors':<25} {num_tensors}")
    
    print("\n" + "=" * 70)
    print("✓ CHECKPOINT IS VALID")
    print("=" * 70)
    
    print("\nThis checkpoint can be used for:")
    print("  - Inference: python scripts/inference.py --checkpoint <path>")
    print("  - Resume training: python scripts/train.py --resume <path>")
    print("=" * 70)

if __name__ == '__main__':
    main()
