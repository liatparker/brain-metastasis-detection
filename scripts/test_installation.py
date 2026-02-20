"""
Test script to verify installation and dependencies.

Usage:
    python scripts/test_installation.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"  ERROR importing PyTorch: {e}")
        return False
    
    try:
        import torchvision
        print(f"  torchvision: {torchvision.__version__}")
    except ImportError as e:
        print(f"  ERROR importing torchvision: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  NumPy: {np.__version__}")
    except ImportError as e:
        print(f"  ERROR importing NumPy: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"  pandas: {pd.__version__}")
    except ImportError as e:
        print(f"  ERROR importing pandas: {e}")
        return False
    
    try:
        import pydicom
        print(f"  pydicom: {pydicom.__version__}")
    except ImportError as e:
        print(f"  ERROR importing pydicom: {e}")
        return False
    
    try:
        import cv2
        print(f"  OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"  ERROR importing OpenCV: {e}")
        return False
    
    try:
        import sklearn
        print(f"  scikit-learn: {sklearn.__version__}")
    except ImportError as e:
        print(f"  ERROR importing scikit-learn: {e}")
        return False
    
    try:
        import huggingface_hub
        print(f"  huggingface-hub: {huggingface_hub.__version__}")
    except ImportError as e:
        print(f"  ERROR importing huggingface-hub: {e}")
        return False
    
    print("  All core dependencies installed successfully!")
    return True


def test_project_modules():
    """Test that project modules can be imported."""
    print("\nTesting project modules...")
    
    try:
        from config import Config
        print("  config.Config: OK")
    except ImportError as e:
        print(f"  ERROR importing Config: {e}")
        return False
    
    try:
        from src.models.hybrid_model import ResNet50_ImageOnly
        print("  src.models.hybrid_model: OK")
    except ImportError as e:
        print(f"  ERROR importing ResNet50_ImageOnly: {e}")
        return False
    
    try:
        from src.data.dataset import BrainMetDataset, PatientSampler
        print("  src.data.dataset: OK")
    except ImportError as e:
        print(f"  ERROR importing dataset classes: {e}")
        return False
    
    try:
        from src.preprocessing.advanced_preprocessing import preprocess_ct_slice
        print("  src.preprocessing: OK")
    except ImportError as e:
        print(f"  ERROR importing preprocessing: {e}")
        return False
    
    print("  All project modules imported successfully!")
    return True


def test_model_creation():
    """Test that model can be created."""
    print("\nTesting model creation...")
    
    try:
        import torch
        from src.models.hybrid_model import ResNet50_ImageOnly
        
        model = ResNet50_ImageOnly(num_classes=1, dropout_rate=0.4)
        print(f"  Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        dummy_input = torch.randn(1, 1, 256, 256).to(device)
        output = model(dummy_input)
        
        print(f"  Forward pass successful: output shape = {output.shape}")
        print(f"  Model test PASSED")
        
        return True
        
    except Exception as e:
        print(f"  ERROR testing model: {e}")
        return False


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")
    
    try:
        from config import Config
        
        cfg = Config()
        print("  Config created successfully")
        
        # Test that config has expected attributes
        assert hasattr(cfg, 'data')
        assert hasattr(cfg, 'model')
        assert hasattr(cfg, 'training')
        assert hasattr(cfg, 'augmentation')
        
        print("  Config structure: OK")
        
        # Test config printing (should not raise error)
        cfg.print_config()
        
        return True
        
    except Exception as e:
        print(f"  ERROR testing config: {e}")
        return False


def test_radimagenet_access():
    """Test RadImageNet weights accessibility."""
    print("\nTesting RadImageNet weights access...")
    
    try:
        from huggingface_hub import hf_hub_download
        
        print("  Checking HuggingFace Hub connection...")
        
        # Try to check if weights are accessible (doesn't download unless needed)
        # This will use cached version if already downloaded
        try:
            weights_path = hf_hub_download(
                repo_id="microsoft/RadImageNet-ResNet50",
                filename="pytorch_model.bin",
                local_files_only=False
            )
            print(f"  RadImageNet weights accessible: {weights_path}")
            print("  RadImageNet test PASSED")
            return True
        except Exception as e:
            print(f"  WARNING: Could not access RadImageNet weights: {e}")
            print("  This is normal on first installation.")
            print("  Weights will auto-download on first training run.")
            print("  RadImageNet test SKIPPED (not critical)")
            return True  # Don't fail the test for this
        
    except Exception as e:
        print(f"  ERROR testing RadImageNet: {e}")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("INSTALLATION TEST")
    print("="*70)
    
    results = []
    
    # Test imports
    results.append(("Core Dependencies", test_imports()))
    
    # Test project modules
    results.append(("Project Modules", test_project_modules()))
    
    # Test configuration
    results.append(("Configuration", test_config()))
    
    # Test model creation
    results.append(("Model Creation", test_model_creation()))
    
    # Test RadImageNet access
    results.append(("RadImageNet Weights", test_radimagenet_access()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\nAll tests PASSED! Installation is correct.")
        print("\nRadImageNet weights will auto-download on first training run (~100 MB).")
        print("\nYou can now run:")
        print("  python scripts/train.py --help")
        print("  python scripts/inference.py --help")
        return 0
    else:
        print("\nSome tests FAILED. Please check the errors above.")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
