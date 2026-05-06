# src/losses/edl_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EDLLoss(nn.Module):
    """
    Evidential Deep Learning Loss (Sensoy et al., NeurIPS 2018)
    Uses the MSE formulation with KL divergence annealing.
    """
    def __init__(self, num_classes=3, annealing_epochs=10):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_epochs = annealing_epochs

    def forward(self, logits, targets, epoch):
        # 1. Map logits to evidence using ReLU (as per Sensoy et al.)
        evidence = F.relu(logits)
        
        # 2. Calculate Dirichlet parameters (alpha) and total strength (S)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # 3. Expected probabilities
        p = alpha / S
        
        # 4. Create one-hot targets
        y = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # 5. Calculate MSE Loss (Error + Variance terms)
        err = torch.sum((y - p)**2, dim=1, keepdim=True)
        var = torch.sum(p * (1 - p) / (S + 1), dim=1, keepdim=True)
        
        # 6. KL Divergence penalty for out-of-class evidence
        annealing_coef = min(1.0, epoch / self.annealing_epochs)
        alpha_tilde = (alpha - 1) * (1 - y) + 1
        
        # Simplified KL divergence for uniform prior
        kl = annealing_coef * torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
        
        loss = err + var + kl
        return torch.mean(loss)

def get_edl_uncertainty(logits):
    """Calculates Epistemic Uncertainty (u) from EDL logits."""
    evidence = F.relu(logits)
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    num_classes = logits.shape[1]
    u = num_classes / S  # Uncertainty is K / S
    return u