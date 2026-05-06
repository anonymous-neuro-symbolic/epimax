import torch
import torch.nn as nn

class BipolarPossibilisticLoss(nn.Module):
    """
    Bipolar possibilistic focal loss for multi-class classification and logical strong negation.
    Formulated to enforce epistemic sufficiency and active rejection manifolds.
    """
    def __init__(self, gamma=2.0, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, alpha=10.0, eps=1e-7):
        super(BipolarPossibilisticLoss, self).__init__()
        self.gamma = gamma
        self.l1 = lambda_1
        self.l2 = lambda_2
        self.l3 = lambda_3
        self.alpha = alpha
        self.eps = eps

    def forward(self, pi_plus, pi_minus, target):
        """
        Args:
            pi_plus (Tensor): Positive possibility outputs (batch_size, num_classes) bounded [0, 1].
            pi_minus (Tensor): Negative possibility outputs (batch_size, num_classes) bounded [0, 1].
            target (Tensor): Ground truth labels (batch_size), integer indices.
        """
        batch_size, num_classes = pi_plus.shape
        
        # Clamp for numerical stability to prevent log(0)
        pi_plus = torch.clamp(pi_plus, self.eps, 1.0 - self.eps)
        pi_minus = torch.clamp(pi_minus, self.eps, 1.0 - self.eps)

        # 1. Isolate the target (ground truth) and alternatives
        # Create a one-hot mask for the target
        mask_target = torch.zeros_like(pi_plus).scatter_(1, target.unsqueeze(1), 1.0)
        mask_alt = 1.0 - mask_target

        # Extract pi_plus for the ground truth class
        pi_gt = (pi_plus * mask_target).sum(dim=1)  # shape: (batch_size,)

        # 2. Compute Necessity (N)
        # N = 1 - max(pi_plus of all alternative classes)
        # We multiply by mask_alt so the target class becomes 0 and doesn't affect the max
        pi_alt_max, _ = (pi_plus * mask_alt).max(dim=1)
        N_gt = 1.0 - pi_alt_max
        N_gt = torch.clamp(N_gt, self.eps, 1.0 - self.eps)

        # 3. Consistency head (L_plus)
        term_cons = self.l1 * ((1.0 - pi_gt) ** self.gamma) * torch.log(pi_gt)
        term_nec = self.l2 * ((1.0 - N_gt) ** self.gamma) * torch.log(N_gt)
        L_plus = -(term_cons + term_nec)

        # 4. Strong negation head (L_minus)
        # We only apply strong negation to alternative classes (y != y*)
        # Sum over all alternative classes: (1 - pi_minus)^gamma * log(pi_minus)
        L_minus_matrix = self.l3 * ((1.0 - pi_minus) ** self.gamma) * torch.log(pi_minus)
        L_minus = -(L_minus_matrix * mask_alt).sum(dim=1)

        # 5. Consistency constraint (L_C)
        # max(0, pi_plus + pi_minus - 1)^2 sum over all classes
        conflict = torch.relu(pi_plus + pi_minus - 1.0)
        L_C = self.alpha * (conflict ** 2).sum(dim=1)

        # Total Loss (mean over batch)
        total_loss = (L_plus + L_minus + L_C).mean()

        return total_loss, {
            "L_plus": L_plus.mean().item(),
            "L_minus": L_minus.mean().item(),
            "L_C": L_C.mean().item(),
            "mean_N": N_gt.mean().item() # Useful for tracking epistemic convergence
        }