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

class JointQSupremumLoss(nn.Module):
    """
    Integration A: The global lattice manifold.
    Jointly optimizes local Possibility and global Necessity using a constrained margin.
    """
    def __init__(self, q=0.5, lambda_n=1.0):
        super().__init__()
        self.q = q
        self.lambda_n = lambda_n

    def forward(self, logits, targets):
        # logits are expected to be <= 0 (e.g., from RBF negative distance)
        z_y = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # 1. Local Possibility Loss: L_pi = -log_q(pi(y)) = -z_y
        L_pi = -z_y
        
        # 2. Global Necessity Loss: L_N = -log_q(N(y))
        mask = torch.ones_like(logits).scatter_(1, targets.unsqueeze(1), 0.0)
        pi_neg = q_exp(logits, self.q) * mask
        
        # S = sum(exp_q(z_j))
        S = torch.sum(pi_neg, dim=1)
        
        # pi(y^c) = log_q(S). We clamp at 0 because possibility cannot be negative.
        pi_y_c = F.relu(q_log(S, self.q))
        
        # N(y) = 1 - pi(y^c). Clamped at 1e-6 to prevent NaN in outer q_log.
        N_y = torch.clamp(1.0 - pi_y_c, min=1e-6)
        L_N = -q_log(N_y, self.q)
        
        # 3. Joint Constrained Optimization
        # The penalty is only active if Necessity is lagging behind Possibility
        constraint_penalty = F.relu(L_N - L_pi)
        loss = L_pi + self.lambda_n * constraint_penalty
        
        return loss.mean()

class JointQBinaryLoss(nn.Module):
    """
    Integration B: The independent q-binary matrix.
    Jointly optimizes local Possibility and independent Necessity evaluations.
    """
    def __init__(self, q=0.5, lambda_n=1.0, margin=0.0):
        super().__init__()
        self.q = q
        self.lambda_n = lambda_n
        self.m = margin

    def forward(self, logits, targets):
        batch_size, num_classes = logits.shape
        z_y = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # 1. Local Possibility Loss
        L_pi = -z_y
        
        # 2. Independent Necessity Loss
        pi = q_exp(logits, self.q)
        mask = torch.ones_like(logits).scatter_(1, targets.unsqueeze(1), 0.0)
        pi_neg = pi * mask
        
        # N_ind(y) requires independently pushing down all pi_j
        safe_complement = torch.clamp(1.0 - pi_neg, min=1e-6)
        penalty_per_class = F.relu(self.m - q_log(safe_complement, self.q))
        
        # Average the necessity penalty over the K-1 complement classes
        L_N = torch.sum(penalty_per_class * mask, dim=1) / (num_classes - 1)
        
        # 3. Additive Joint Optimization (valid due to independent complement constraints)
        loss = L_pi + self.lambda_n * L_N
        
        return loss.mean()