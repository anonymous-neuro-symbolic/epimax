import torch
import torch.nn as nn
import torch.nn.functional as F

def q_exp(x, q):
    """Tsallis q-exponential function. Compact support when q < 1."""
    base = F.relu(1.0 + (1.0 - q) * x)
    return torch.pow(base + 1e-8, 1.0 / (1.0 - q))

def q_log(x, q):
    """Tsallis q-logarithm function. Inverse of q_exp."""
    return (torch.pow(x + 1e-8, 1.0 - q) - 1.0) / (1.0 - q)

class QSupremumMarginLoss(nn.Module):
    def __init__(self, q=0.5, margin=1.0, apply_relative_anchor=False):
        super().__init__()
        self.q = q
        self.m = margin
        self.apply_relative_anchor = apply_relative_anchor

    def forward(self, logits, targets):
        # 1. THE RELATIVE ANCHOR: z = z - max(z)
        # This guarantees the highest logit is exactly 0.0, bounding pi to exactly 1.0
        if self.apply_relative_anchor:
            logits = logits - logits.max(dim=1, keepdim=True)[0]

        z_y = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        mask = torch.ones_like(logits).scatter_(1, targets.unsqueeze(1), 0.0)
        
        q_exp_neg = q_exp(logits, self.q) * mask
        sum_q_exp = torch.sum(q_exp_neg, dim=1)
        S_q = q_log(sum_q_exp, self.q)
        
        loss = F.relu(self.m + S_q - z_y)
        return loss.mean()

class QBinaryPossibilisticLoss(nn.Module):
    def __init__(self, q=0.5, margin=2.0, apply_relative_anchor=False):
        super().__init__()
        self.q = q
        self.m = margin
        self.apply_relative_anchor = apply_relative_anchor

    def forward(self, logits, targets):
        # 1. THE RELATIVE ANCHOR (if using standard MLP)
        if self.apply_relative_anchor:
            logits = logits - logits.max(dim=1, keepdim=True)[0]

        batch_size, num_classes = logits.shape
        pi = q_exp(logits, self.q)
        
        pi_y = pi.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss_pos = -q_log(pi_y, self.q)
        
        mask = torch.ones_like(logits).scatter_(1, targets.unsqueeze(1), 0.0)
        pi_neg = pi * mask
        
        safe_complement = torch.clamp(1.0 - pi_neg, min=1e-6)
        penalty = F.relu(self.m - q_log(safe_complement, self.q))
        loss_neg = torch.sum(penalty * mask, dim=1) / (num_classes - 1)
        
        return (loss_pos + loss_neg).mean()