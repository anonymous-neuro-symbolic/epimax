import torch
import torch.nn.functional as F

def smooth_min(a, b, tau=10.0):
    """
    Differentiable logical AND.
    Uses temperature-scaled softmin weighting to preserve the [0, 1] bounds.
    """
    stacked = torch.stack([a, b], dim=0)
    # Negative tau weights the smaller value closer to 1.0
    weights = F.softmax(-tau * stacked, dim=0)
    return (weights * stacked).sum(dim=0)

def smooth_max(tensor_list, tau=10.0):
    """
    Differentiable logical OR.
    Uses temperature-scaled softmax weighting to preserve the [0, 1] bounds.
    """
    stacked = torch.stack(tensor_list, dim=0)
    # Positive tau weights the larger value closer to 1.0
    weights = F.softmax(tau * stacked, dim=0)
    return (weights * stacked).sum(dim=0)

def probabilistic_addition(prob_list):
    """
    DeepProbLog baseline: Sum-Product algebra.
    Autograd-safe (Out-of-place) implementation.
    """
    current_dist = prob_list[0]
    
    for next_prob in prob_list[1:]:
        current_max = current_dist.size(1)
        next_max = next_prob.size(1)
        new_max = current_max + next_max - 1
        
        new_dist_elements = []
        
        for k in range(new_max):
            k_probs = []
            for i in range(current_max):
                j = k - i
                if 0 <= j < next_max:
                    k_probs.append(current_dist[:, i] * next_prob[:, j])
            
            if k_probs:
                new_dist_elements.append(torch.stack(k_probs, dim=0).sum(dim=0))
            else:
                new_dist_elements.append(torch.zeros_like(current_dist[:, 0]))
                
        current_dist = torch.stack(new_dist_elements, dim=1)
        
    return current_dist

def possibilistic_addition(pi_list, tau=10.0):
    """
    EpiMax framework: Smooth Max-Min algebra.
    Autograd-safe (Out-of-place) implementation with differentiable routing.
    """
    current_pi = pi_list[0]
    
    for next_pi in pi_list[1:]:
        current_max = current_pi.size(1)
        next_max = next_pi.size(1)
        new_max = current_max + next_max - 1
        
        new_pi_elements = []
        
        for k in range(new_max):
            k_conjunctions = []
            for i in range(current_max):
                j = k - i
                if 0 <= j < next_max:
                    # Differentiable AND
                    k_conjunctions.append(smooth_min(current_pi[:, i], next_pi[:, j], tau=tau))
            
            if k_conjunctions:
                # Differentiable OR
                new_pi_elements.append(smooth_max(k_conjunctions, tau=tau))
            else:
                new_pi_elements.append(torch.zeros_like(current_pi[:, 0]))
                
        current_pi = torch.stack(new_pi_elements, dim=1)
        
    return current_pi