"""
ResNet-50 model for brain metastasis detection.

Main Model:
- ResNet50_ImageOnly: Single-channel CT input for image-only training
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet50_ImageOnly(nn.Module):
    """
    Image-only ResNet-50 model for brain metastasis detection.
    
    Architecture:
    - Input: Single-channel CT slice [B, 1, 256, 256]
    - Backbone: ResNet-50 pretrained on RadImageNet
    - Classifier: Linear(2048 → 512) → ReLU → Dropout → Linear(512 → 1)
    - Output: Logit [B, 1] (use sigmoid for probability)
    
    This is the main model used for training in the image-only pipeline.
    """
    
    def __init__(self, num_classes=1, dropout_rate=0.4):
        """
        Initialize ResNet50 image-only model.
        
        Args:
            num_classes: Number of output classes (1 for binary classification)
            dropout_rate: Dropout probability in classifier (default: 0.4)
        """
        super(ResNet50_ImageOnly, self).__init__()
        
        # Load pretrained ResNet-50 (ImageNet by default)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Modify first conv layer for single-channel input
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(
            in_channels=1,  # Single channel CT
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )
        
        # Initialize by averaging RGB weights
        with torch.no_grad():
            self.conv1.weight = nn.Parameter(
                original_conv1.weight.mean(dim=1, keepdim=True)
            )
        
        # Extract backbone features (conv1 through layer4 + avgpool)
        self.backbone = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # Index 4
            resnet.layer2,  # Index 5
            resnet.layer3,  # Index 6
            resnet.layer4,  # Index 7
            resnet.avgpool  # Index 8
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Store references for easier access
        self.layer3 = self.backbone[6]
        self.layer4 = self.backbone[7]
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 1, H, W]
        
        Returns:
            Logit tensor [B, 1]
        """
        # Extract features
        features = self.backbone(x)
        features = torch.flatten(features, 1)  # [B, 2048]
        
        # Classify
        output = self.classifier(features)  # [B, 1]
        
        return output


def load_radimagenet_weights(model):
    """
    Load RadImageNet pretrained weights from HuggingFace.
    
    Downloads and loads RadImageNet-ResNet50 weights, which are pretrained
    on 1.35M medical images (better than ImageNet for CT scans).
    
    Args:
        model: ResNet50_ImageOnly model instance
    
    Note:
        Requires huggingface_hub to be installed.
        Falls back to ImageNet weights if download fails.
    """
    try:
        from huggingface_hub import hf_hub_download
        
        print("Downloading RadImageNet weights from HuggingFace...")
        weights_path = hf_hub_download(
            repo_id="microsoft/RadImageNet-ResNet50",
            filename="pytorch_model.bin"
        )
        
        # Load weights
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Load into backbone (skip first conv since we modified it)
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in state_dict.items() 
            if k in model_dict and 'conv1' not in k
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        
        print("✓ RadImageNet weights loaded successfully")
        
    except Exception as e:
        print(f"⚠ Failed to load RadImageNet weights: {e}")
        print("  Using ImageNet weights instead")


