import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class TabularCentroidNet(nn.Module):
    """
    MLP architecture with Spectral Normalization and a Centroid Head.
    Designed for heterogeneous tabular clinical features.
    """
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64, num_classes=2):
        super(TabularCentroidNet, self).__init__()
        
        # Spectral-normalized MLP backbone
        self.backbone = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, latent_dim)),
            nn.ReLU()
        )
        
        # Learnable centroids in the latent space
        self.centroids = nn.Parameter(torch.randn(num_classes, latent_dim))

    def forward(self, x):
        features = self.backbone(x)
        # Compute negative squared Euclidean distance: z = -||f(x) - c||^2
        dist = torch.cdist(features, self.centroids, p=2) ** 2
        return -dist