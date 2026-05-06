import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils import spectral_norm

class CentroidHead(nn.Module):
    """
    Replaces the standard linear layer. 
    Computes negative squared Euclidean distance to learnable class centroids.
    Guarantees z <= 0, topologically anchoring the possibility pi to [0, 1].
    """
    def __init__(self, in_features, num_classes):
        super(CentroidHead, self).__init__()
        # Initialize K class centroids in the latent feature space
        self.centroids = nn.Parameter(torch.randn(num_classes, in_features))

    def forward(self, x):
        # Calculate negative squared Euclidean distance: -||f(x) - c||^2
        dist = torch.cdist(x, self.centroids, p=2) ** 2
        return -dist

def apply_spectral_normalization(module):
    """
    Recursively applies spectral normalization to all convolutional 
    and linear layers within a given module to enforce bi-Lipschitz continuity.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            # Wrap the layer with PyTorch's native spectral_norm
            setattr(module, name, spectral_norm(child))
        else:
            # Recursively apply to sub-modules (like BasicBlock in ResNet)
            apply_spectral_normalization(child)

def create_centroid_resnet18(num_classes=4, enforce_lipschitz=True):
    """
    Constructs a ResNet-18 tailored for CIFAR-10 with a Distance-to-Centroid head.
    """
    model = models.resnet18(weights=None)
    
    # Adapt ResNet-18 for 32x32 CIFAR-10 images (remove 7x7 conv and maxpool)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    # Apply spectral normalization to the feature extractor to prevent space collapse
    if enforce_lipschitz:
        apply_spectral_normalization(model)
    
    # Replace the fully connected layer with the Centroid Head
    in_features = model.fc.in_features
    model.fc = CentroidHead(in_features=in_features, num_classes=num_classes)
    
    return model