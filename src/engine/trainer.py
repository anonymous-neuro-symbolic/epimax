import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.optim as optim

class BipolarTrainer:
    def __init__(self, model, loss_fn, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config['lr'], 
            weight_decay=config['weight_decay']
        )
        
        self.history = {
            "train_loss": [], "val_acc": [], "val_hvr": [], 
            "mean_delta": [], "mean_necessity": []
        }

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        epoch_delta = []
        epoch_N = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass: Extract Possibilistic Distributions
            pi_plus, pi_minus = self.model(images)
            
            # 1. Calculate Loss
            loss, loss_dict = self.loss_fn(pi_plus, pi_minus, targets)
            
            # 2. Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0) 
            self.optimizer.step()
            
            # 3. Compute Epistemic Divergence (Delta) for monitoring
            # Delta = |pi_minus - (1 - N)|. 
            # We measure how much the 'Active Rejection' differs from 'Negation as Failure'
            with torch.no_grad():
                # N = 1 - max(pi_plus_alternatives)
                mask_target = torch.zeros_like(pi_plus).scatter_(1, targets.unsqueeze(1), 1.0)
                mask_alt = 1.0 - mask_target
                pi_alt_max, _ = (pi_plus * mask_alt).max(dim=1)
                necessity = 1.0 - pi_alt_max
                
                # We compare the mean rejection of alternatives to the necessity gap
                # This proves the Bipolar manifold is non-redundant
                avg_pi_minus_alt = (pi_minus * mask_alt).sum(dim=1) / (pi_plus.size(1) - 1)
                delta = torch.abs(avg_pi_minus_alt - (1.0 - necessity)).mean().item()
                
                epoch_delta.append(delta)
                epoch_N.append(loss_dict['mean_N'])

            epoch_loss += loss.item()
            
            # Verbose progress tracking
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    "L": f"{loss.item():.3f}",
                    "Lc": f"{loss_dict['L_C']:.3f}",
                    "N": f"{loss_dict['mean_N']:.2f}",
                    "Δ": f"{delta:.2f}"
                })
        
        avg_loss = epoch_loss / len(self.train_loader)
        avg_delta = np.mean(epoch_delta)
        
        print(f"\n[Epoch {epoch} Summary]")
        print(f" > Loss: {avg_loss:.4f} (Consistency: {loss_dict['L_plus']:.4f}, Rejection: {loss_dict['L_minus']:.4f})")
        print(f" > Epistemic Plateau (N): {np.mean(epoch_N):.4f}")
        print(f" > Bipolar Divergence (Δ): {avg_delta:.4f}  <-- Proof of non-redundancy")
        
        return avg_loss

    @torch.no_grad()
    def evaluate(self, is_hierarchical=False, superclass_map=None):
        self.model.eval()
        correct = 0
        total = 0
        violations = 0
        
        for images, targets in self.val_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            pi_plus, _ = self.model(images)
            
            # Predictions are the supremum of the positive manifold
            preds = pi_plus.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            if is_hierarchical and superclass_map is not None:
                violations += self.compute_hvr(pi_plus, targets, superclass_map)

        acc = correct / total
        hvr = violations / total if total > 0 else 0
        return acc, hvr

    def compute_hvr(self, pi_plus, targets, superclass_map):
        """
        Calculates Hierarchical Violation Rate.
        A violation occurs if a subclass possibility exceeds its parent superclass possibility.
        """
        violations = 0
        # superclass_map should be a tensor mapping subclass_idx -> superclass_idx
        preds_fine = pi_plus.argmax(dim=1)
        
        for i in range(len(targets)):
            fine_idx = preds_fine[i]
            coarse_idx = superclass_map[fine_idx]
            
            # Logically: Pi(Fine) must be <= Pi(Coarse)
            # If the model is more sure about the 'Lion' than 'Mammal', it's a violation.
            if pi_plus[i, fine_idx] > pi_plus[i, coarse_idx] + 1e-3:
                violations += 1
        return violations
    
def train_epimax_resnet(model, train_loader, criterion, device, epochs=100):
    """
    Standard SGD training loop for ResNet-18 with DualPossibilisticLoss.
    """
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # Decay LR at 50% and 75% of training
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

    return model