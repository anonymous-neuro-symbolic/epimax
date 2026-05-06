import torch
import torch.nn.functional as F
import torch.optim as optim
import time

from src.datasets.mnist_nesy import get_nesy_loaders
from src.models.nesy_cnn import NeSyCNN
from src.engine.logic_engine import probabilistic_addition, possibilistic_addition

# Experimental constraints
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 3          # 3 epochs is sufficient for MNIST-Addition convergence
BATCH_SIZE = 128
LR = 0.001
GAMMA = 2.0         # Focal pressure for the NeSy possibilistic loss

def train_probabilistic(model, train_loader, optimizer, epoch):
    """DeepProbLog Baseline: Cross-Entropy over Sum-Product logic"""
    model.train()
    total_loss = 0
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        
        # 1. Perception: Extract probabilities for each digit
        prob_list = []
        for i in range(images.size(1)):
            logits = model(images[:, i])
            prob_list.append(F.softmax(logits, dim=1))
            
        # 2. Reasoning: Sum-Product algebraic aggregation
        prob_sum = probabilistic_addition(prob_list)
        
        # 3. Learning: Standard Negative Log-Likelihood
        loss = F.nll_loss(torch.log(prob_sum + 1e-7), targets)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"[Baseline] Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f}")

def train_possibilistic(model, train_loader, optimizer, epoch):
    """EpiMax Framework: Dual Possibilistic Loss over Max-Min logic"""
    model.train()
    total_loss = 0
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        
        # 1. Perception: Extract possibilities (pi) for each digit
        pi_list = []
        for i in range(images.size(1)):
            logits = model(images[:, i])
            z_max, _ = torch.max(logits, dim=1, keepdim=True)
            pi = torch.clamp(torch.exp(logits - z_max), 1e-7, 1.0)
            pi_list.append(pi)
            
        # 2. Reasoning: Max-Min algebraic aggregation
        pi_sum = possibilistic_addition(pi_list)
        
        # 3. Learning: Neuro-Symbolic Dual Possibilistic Loss
        batch_size = targets.size(0)
        pi_target = torch.clamp(pi_sum[torch.arange(batch_size), targets], 1e-7, 1.0)
        
        mask = torch.ones_like(pi_sum, dtype=torch.bool)
        mask[torch.arange(batch_size), targets] = False
        pi_sum_without_target = pi_sum.masked_fill(~mask, 0.0)
        n_bar_target, _ = torch.max(pi_sum_without_target, dim=1)
        
        focal_base = torch.clamp(1.0 - pi_target, min=1e-7)
        loss = torch.pow(focal_base, GAMMA) + torch.pow(n_bar_target, GAMMA)
        loss = loss.mean()
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[EpiMax]   Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f}")

def evaluate_logic(model, loader, mode='probabilistic'):
    """Evaluates the aggregated accuracy of the logical chain"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            # Extract independent outputs
            outputs_list = []
            for i in range(images.size(1)):
                logits = model(images[:, i])
                if mode == 'probabilistic':
                    outputs_list.append(F.softmax(logits, dim=1))
                else:
                    z_max, _ = torch.max(logits, dim=1, keepdim=True)
                    outputs_list.append(torch.clamp(torch.exp(logits - z_max), 1e-7, 1.0))
            
            # Aggregate based on logic framework
            if mode == 'probabilistic':
                final_dist = probabilistic_addition(outputs_list)
            else:
                final_dist = possibilistic_addition(outputs_list)
                
            predictions = final_dist.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            
    return (correct / total) * 100.0

def main():
    print(f"Initializing Neuro-Symbolic logic experiment on {DEVICE}...")
    train_loader, eval_loaders = get_nesy_loaders(batch_size=BATCH_SIZE, train_digits=2)
    
    # Initialize separate models for strict isolation
    model_prob = NeSyCNN().to(DEVICE)
    model_poss = NeSyCNN().to(DEVICE)
    
    opt_prob = optim.Adam(model_prob.parameters(), lr=LR)
    opt_poss = optim.Adam(model_poss.parameters(), lr=LR)
    
    # 1. Training Phase (2-digit addition only)
    print("\n--- Training Phase (2-Digit Addition) ---")
    for epoch in range(1, EPOCHS + 1):
        train_probabilistic(model_prob, train_loader, opt_prob, epoch)
        train_possibilistic(model_poss, train_loader, opt_poss, epoch)
        
    # 2. Evaluation Phase (Extrapolating to N-digits)
    print("\n--- Multi-Hop Extrapolation Evaluation ---")
    print(f"{'Chain Length':<15} | {'DeepProbLog (Prob)':<20} | {'EpiMax (Poss)':<20}")
    print("-" * 60)
    
    for n_digits, loader in eval_loaders.items():
        acc_prob = evaluate_logic(model_prob, loader, mode='probabilistic')
        acc_poss = evaluate_logic(model_poss, loader, mode='possibilistic')
        print(f"{n_digits}-Digit Add    | {acc_prob:>18.2f}% | {acc_poss:>18.2f}%")

if __name__ == "__main__":
    main()