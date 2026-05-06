import torch
from .epistemic_losses import _get_competitor_possibility

def evaluate_chain_necessity(logits_list, targets_list):
    """
    Evaluates a logical conjunction across a chain of length L using only 
    the EpiMax (possibilistic) framework to demonstrate epistemic stability.
    """
    pi_nots = []
    for logits, target in zip(logits_list, targets_list):
        pi_nots.append(_get_competitor_possibility(logits, target))
    
    # Apply the weakest link principle: N(C) = 1 - max(\Pi(\neg A_k))
    pi_nots_tensor = torch.stack(pi_nots, dim=0)
    max_pi_not, _ = torch.max(pi_nots_tensor, dim=0)
    poss_conf = 1.0 - max_pi_not 
            
    return poss_conf