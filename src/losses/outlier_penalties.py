import torch
import torch.nn.functional as F
from src.losses.joint_q_losses import q_exp

def active_repulsive_loss(logits, q=0.5):
    """
    Computes the active repulsive penalty for outlier exposure.
    Forces the sum of possibilities across all classes toward zero for background data.
    """
    # Map negative distances (logits) to the unit interval
    pi = q_exp(logits, q)
    
    # Minimize the total possibility of the sample belonging to any known class
    # This effectively 'hollows out' the manifold for OOD data
    loss_oe = torch.sum(pi, dim=1)
    
    return loss_oe.mean()