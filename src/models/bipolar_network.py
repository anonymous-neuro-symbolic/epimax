
import os
import torch
import torch.nn as nn
import torchvision.models as models

class EnergyToPossibility(nn.Module):
    """
    Transforms raw logits (negative energies) into a qualitative 
    possibilistic distribution natively satisfying the consistency axiom.
    Mapping: pi(y) = exp( beta * (z_y - max(z)) )
    """
    def __init__(self, initial_beta=1.0, learnable=True):
        super(EnergyToPossibility, self).__init__()
        if learnable:
            # We initialize beta as a parameter. Using a log-space 
            # parameterization ensures beta stays positive.
            # self.log_beta = nn.Parameter(torch.tensor(torch.log(torch.tensor(initial_beta))))
            self.log_beta = nn.Parameter(torch.log(torch.tensor(float(initial_beta))))
        else:
            self.register_buffer('log_beta', torch.tensor(torch.log(torch.tensor(initial_beta))))

    def forward(self, logits):
        beta = torch.exp(self.log_beta)
        
        # 1. Identify the minimum energy state (maximum logit) per sample
        # max_logits shape: (batch_size, 1)
        max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
        
        # 2. Compute the relative energy gap (z - max(z))
        # This is always <= 0, ensuring exp(...) is in (0, 1]
        energy_gap = logits - max_logits
        
        # 3. Apply the max-normalized exponential mapping
        pi = torch.exp(beta * energy_gap)
        
        return pi

class BipolarPossibilisticNetwork(nn.Module):
    """
    Wrapper that transforms a backbone into a dual-manifold 
    energy-based possibilistic model.
    """
    def __init__(self, backbone, feature_dim, num_pos_classes, num_neg_classes=None):
        super(BipolarPossibilisticNetwork, self).__init__()
        
        if num_neg_classes is None:
            num_neg_classes = num_pos_classes
            
        self.backbone = backbone
        
        # Logit Generators (Energy Landscape mappings)
        self.head_plus = nn.Linear(feature_dim, num_pos_classes)
        self.head_minus = nn.Linear(feature_dim, num_neg_classes)
        
        # Possibilistic Mappings (Alternative 2: Learnable Beta)
        self.pos_mapping = EnergyToPossibility(learnable=True)
        self.neg_mapping = EnergyToPossibility(learnable=True)

    def forward(self, x):
        # Extract latent features from the backbone
        features = self.backbone(x)
        
        # Generate raw energy potentials (logits)
        logits_plus = self.head_plus(features)
        logits_minus = self.head_minus(features)
        
        # Apply L_inf normalization to reach possibility space [0, 1]
        # This ensures sup(pi_plus) = 1 and sup(pi_minus) = 1
        pi_plus = self.pos_mapping(logits_plus)
        pi_minus = self.neg_mapping(logits_minus)
        
        return pi_plus, pi_minus

def create_bipolar_model(model_name: str, num_pos: int, num_neg: int = None, pretrained: bool = True):
    """
    Modified to support both the V100 high-compute model and the laptop MNIST model.
    """
    if model_name == 'wide_resnet50_2':
        # (Your existing code for the V100)
        backbone = models.wide_resnet50_2(weights=None)
        if pretrained:
            local_path = "wide_resnet50_2-9ba9bcbe.pth"
            if os.path.exists(local_path):
                print(f"Loading local weights from: {local_path}")
                state_dict = torch.load(local_path, map_location='cpu')
                backbone.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(f"Pretrained weights not found at {local_path}.")
        
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity() 

    elif model_name == 'resnet18_mnist':
        # --- NEW: Lightweight ResNet for MNIST (Laptop OOD test) ---
        backbone = models.resnet18(weights=None)
        # Adapt for 1-channel grayscale images (MNIST) instead of 3-channel RGB
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

    elif model_name == 'resnet18_cifar':
            backbone = models.resnet18(weights=None)
            # CIFAR-10 images are 32x32, so we remove the initial aggressive downsampling
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            backbone.maxpool = nn.Identity()
            
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()        

    else:
        raise ValueError(f"Architecture {model_name} not supported.")

    return BipolarPossibilisticNetwork(backbone, feature_dim, num_pos, num_neg)