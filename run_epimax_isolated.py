import torch
import torch.optim as optim

from src.datasets.mnist_nesy import get_nesy_loaders
from src.models.nesy_cnn import NeSyCNN
from src.engine.logic_engine import possibilistic_addition

# Experimental constraints
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 15          # Extended training budget
BATCH_SIZE = 128
LR = 0.001
GAMMA = 2.0         
TAU = 5.0            # Slightly lower temperature to encourage wider gradient flow early on

def evaluate_logic(model, loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            outputs_list = []
            for i in range(images.size(1)):
                logits = model(images[:, i])
                z_max, _ = torch.max(logits, dim=1, keepdim=True)
                outputs_list.append(torch.clamp(torch.exp(logits - z_max), 1e-7, 1.0))
            
            final_dist = possibilistic_addition(outputs_list, tau=TAU)
            predictions = final_dist.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            
    return (correct / total) * 100.0

def main():
    print(f"Initializing Isolated EpiMax experiment on {DEVICE}...")
    # Get loaders (using 10-digit for the final extrapolation test)
    train_loader, eval_loaders = get_nesy_loaders(batch_size=BATCH_SIZE, train_digits=2, eval_digits=[2, 10])
    
    model = NeSyCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("\n--- Training Phase (2-Digit Addition) ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        
        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            
            pi_list = []
            for i in range(images.size(1)):
                logits = model(images[:, i])
                z_max, _ = torch.max(logits, dim=1, keepdim=True)
                pi = torch.clamp(torch.exp(logits - z_max), 1e-7, 1.0)
                pi_list.append(pi)
                
            pi_sum = possibilistic_addition(pi_list, tau=TAU)
            
            # Dual Possibilistic Loss
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

        # Mid-training evaluation to track the learning curve
        val_acc = evaluate_logic(model, eval_loaders[2])
        print(f"Epoch {epoch:<2} | Loss: {total_loss/len(train_loader):.4f} | 2-Digit Acc: {val_acc:.2f}%")

    print("\n--- Final Extrapolation Evaluation ---")
    acc_10 = evaluate_logic(model, eval_loaders[10])
    print(f"10-Digit Addition Accuracy: {acc_10:.2f}%")

if __name__ == "__main__":
    main()