import torch
import torch.nn.functional as F

def dual_possibilistic_loss(logits, targets, alpha=1.0, beta=1.0, gamma=1.0):
    """
    Implements the foundational EpiMax loss over logits (Equation 1).
    L_DP = alpha * (1 - pi(y*))^gamma + beta * (pi(\neg y*))^gamma
    """
    # 1. Normalize by the absolute maximum logit
    max_all, _ = torch.max(logits, dim=-1, keepdim=True)
    
    # 2. Compute Possibility of the target class: \pi(y^*)
    target_logits = logits.gather(1, targets.unsqueeze(1))
    pi_target = torch.exp(target_logits - max_all).squeeze(-1)
    
    # 3. Compute Possibility of the competitors: \Pi(\neg y^*)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, targets.unsqueeze(1), False)
    competitor_logits = logits.masked_fill(~mask, float('-inf'))
    max_competitor, _ = torch.max(competitor_logits, dim=-1, keepdim=True)
    pi_competitors = torch.exp(max_competitor - max_all).squeeze(-1)
    
    # 4. Compute the dual terms
    feasibility_term = alpha * torch.pow(1.0 - pi_target, gamma)
    certainty_term = beta * torch.pow(pi_competitors, gamma)
    
    return (feasibility_term + certainty_term).mean()

def _get_competitor_possibility(logits, target_class):
    """ Helper to extract exactly \Pi(\neg A) """
    max_all, _ = torch.max(logits, dim=-1, keepdim=True)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, target_class.unsqueeze(1), False)
    competitor_logits = logits.masked_fill(~mask, float('-inf'))
    max_competitor, _ = torch.max(competitor_logits, dim=-1, keepdim=True)
    return torch.exp(max_competitor - max_all).squeeze(-1)

def syllogistic_chain_loss(premises_logits_list, premises_targets_list, conclusion_logits, conclusion_target, gamma=1.0):
    """
    Implements the relational inference constraint: \Pi(\neg B) <= \max_k( \Pi(\neg A_k) )
    """
    pi_not_premises = []
    for logits, target in zip(premises_logits_list, premises_targets_list):
        pi_not_premises.append(_get_competitor_possibility(logits, target))
        
    pi_not_premises_tensor = torch.stack(pi_not_premises, dim=0)
    weakest_link_pi_not_A, _ = torch.max(pi_not_premises_tensor, dim=0)
    
    pi_not_B = _get_competitor_possibility(conclusion_logits, conclusion_target)
    
    violation = F.relu(pi_not_B - weakest_link_pi_not_A)
    return torch.pow(violation, gamma).mean()