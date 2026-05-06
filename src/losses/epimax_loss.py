# src/losses/epimax_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EpiMax(nn.Module):
    """
    The EpiMax Layer: A bifurcated activation function that outputs 
    Possibility (pi) and Necessity (N) directly from logits.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        # Find highest and second highest logits
        top2_values, top2_indices = torch.topk(logits, k=2, dim=1)
        z_max = top2_values[:, 0:1]
        z_sec = top2_values[:, 1:2]

        # 1. Compute Possibility (pi) for all classes
        pi = torch.exp(logits - z_max)

        # 2. Compute Necessity (N) strictly for the predicted class
        # Initialize N as zeros, then populate the winner's index
        N = torch.zeros_like(logits)
        winner_necessity = 1.0 - torch.exp(z_sec - z_max)
        N.scatter_(1, top2_indices[:, 0:1], winner_necessity)

        return pi, N

import torch
import torch.nn as nn

class DualPossibilisticLoss(nn.Module):
    """
    Numerically stable Dual Possibilistic Loss with intermediate logging.
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=2.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets, return_metrics=False):
        batch_size = logits.size(0)
        
        # 1. Isolate target and max logits
        z_target = logits[torch.arange(batch_size), targets].unsqueeze(1)
        z_max, _ = torch.max(logits, dim=1, keepdim=True)
        
        # 2. Find the strongest competitor (z_c)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(batch_size), targets] = False
        logits_without_target = logits.masked_fill(~mask, float('-inf'))
        z_c, _ = torch.max(logits_without_target, dim=1, keepdim=True)

        # 3. Map to Possibility and Complementary Necessity (with safety clamps)
        pi_target = torch.clamp(torch.exp(z_target - z_max), self.eps, 1.0)
        n_bar_target = torch.clamp(torch.exp(z_c - z_max), self.eps, 1.0)
       

        # # 4. Compute focal penalties (Clamp the base at 0.0 to prevent NaN with fractional gamma)
        # focal_base = torch.clamp(1.0 - pi_target, min=0.0)
        # 4. Compute focal penalties (Clamp the base at eps to prevent Zero-Gradient Singularity)
        focal_base = torch.clamp(1.0 - pi_target, min=self.eps)
        possibility_penalty = self.alpha * torch.pow(focal_base, self.gamma)
        necessity_penalty = self.beta * torch.pow(n_bar_target, self.gamma)

        # 5. Dual Loss
        loss = possibility_penalty + necessity_penalty
        mean_loss = loss.mean()

        if return_metrics:
            with torch.no_grad():
                metrics = {
                    'pi_target_mean': pi_target.mean().item(),
                    'n_bar_target_mean': n_bar_target.mean().item(),
                    'margin_mean': (z_target - z_c).mean().item() # The "Epistemic Gap"
                }
            return mean_loss, metrics
            
        return mean_loss