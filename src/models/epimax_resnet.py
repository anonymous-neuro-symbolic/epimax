import torch.nn as nn
from torchvision.models import resnet18
from src.losses.epimax_loss import EpiMax

class EpiMaxResNet18(nn.Module):
    """
    Modified ResNet-18 backbone for small-scale images (CIFAR-10).
    """
    def __init__(self, num_classes=10):
        super(EpiMaxResNet18, self).__init__()
        # Initialize standard backbone
        self.backbone = resnet18(num_classes=num_classes)
        
        # Modification for 32x32 images: preserve spatial resolution
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        
        # EpiMax layer for dual possibilistic outputs (pi, N)
        self.epimax = EpiMax()

    def forward(self, x):
        logits = self.backbone(x)
        
        if self.training:
            # During training, return logits for DualPossibilisticLoss
            return logits
        
        # During inference, return logits and the dual manifolds
        pi, N = self.epimax(logits)
        return logits, pi, N