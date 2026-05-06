import torch
import torch.nn as nn
import torchvision.models as models

class SoftmaxBaseline(nn.Module):
    def __init__(self, num_classes=6): # 0-5 for ID
        super(SoftmaxBaseline, self).__init__()
        # Use a lightweight ResNet for the 3050 VRAM
        self.backbone = models.resnet18(weights=None)
        # Adapt for grayscale MNIST
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.backbone(x)