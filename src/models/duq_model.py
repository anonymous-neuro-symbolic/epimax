# src/models/duq_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DUQModel(nn.Module):
    """
    Deterministic Uncertainty Quantification (van Amersfoort et al., ICML 2020).
    Maps inputs to an RBF-based feature space centered on class centroids.
    """
    def __init__(self, input_dim=2, feature_dim=64, num_classes=3, sigma=0.5):
        super().__init__()
        self.sigma = sigma
        self.num_classes = num_classes
        
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, feature_dim)
        )
        
        # Centroids optimized via gradients
        self.centroids = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, x):
        features = self.backbone(x)
        # Compute squared Euclidean distances: ||f(x) - e_c||^2
        distances = torch.cdist(features, self.centroids)
        # Compute RBF kernels (scores)
        rbf_scores = torch.exp(-(distances**2) / (2 * self.sigma**2))
        return rbf_scores

def duq_loss(rbf_scores, targets):
    """Binary Cross Entropy loss for independent RBF class scores."""
    num_classes = rbf_scores.shape[1]
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    return F.binary_cross_entropy(rbf_scores, targets_one_hot)