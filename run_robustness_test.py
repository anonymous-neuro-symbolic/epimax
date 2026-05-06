import torch
import torch.optim as optim
from torchvision import datasets, transforms
from src.models.perception_net import PerceptionNet
from src.engine.logic_engine import probabilistic_addition, possibilistic_addition
from src.utils.generic_epicstemoc_logs_deeprpblog_vs_epimax import generate_epistemic_plots

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def pretrain_perception(model):
    """Standard 1-epoch MNIST training to establish a base accuracy (~98-99%)"""
    print("Pre-training Perception Layer on MNIST...")
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=128, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for img, label in train_loader:
        img, label = img.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(img), label)
        loss.backward()
        optimizer.step()
    print("Pre-training complete.")

def run_extrapolation(model, chain_length, num_samples=100, mode='probabilistic'):
    """Tests logical addition of N digits using the pre-trained model"""
    model.eval()
    # We load images one by one or in a way that we can group them into chains
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=chain_length, shuffle=True, drop_last=True)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= num_samples: break
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # images shape: [chain_length, 1, 28, 28]
            logits = model(images) # shape: [chain_length, 10]
            
            outputs = []
            if mode == 'probabilistic':
                # Split the batch into a list of individual probabilities
                probs = torch.softmax(logits, dim=1)
                outputs = [probs[j:j+1] for j in range(chain_length)]
                final_dist = probabilistic_addition(outputs)
            else:
                # Split the batch into a list of individual possibilities
                z_max, _ = torch.max(logits, dim=1, keepdim=True)
                pi = torch.exp(logits - z_max)
                outputs = [pi[j:j+1] for j in range(chain_length)]
                # Use high TAU for inference to mimic hard Gödel logic
                final_dist = possibilistic_addition(outputs, tau=100.0) 
            
            target_sum = labels.sum().item()
            # final_dist shape is [1, sum_range], so we take index 0
            pred_sum = final_dist.argmax(dim=1)[0].item()
            
            if pred_sum == target_sum: 
                correct += 1
            total += 1
            
    return (correct / total) * 100

def main():
    # 1. Perception remains frozen after initial warm-up
    model = PerceptionNet().to(DEVICE)
    pretrain_perception(model)
    
    # 2. Extended length sweep to find the ultimate breaking point
    lengths = [2, 5, 10, 15, 20]
    # Significant sample volume to reduce variance (StDev proportional to 1/sqrt(N))
    SAMPLES = 1000 
    
    print(f"\nExtrapolation Robustness Test (Samples per length: {SAMPLES})")
    print(f"{'Chain Length':<15} | {'Probabilistic Acc':<20} | {'Possibilistic Acc':<20}")
    print("-" * 60)
    
    results = {"len": [], "prob": [], "poss": []}
    
    for L in lengths:
        # Run probabilistic baseline
        acc_prob = run_extrapolation(model, L, num_samples=SAMPLES, mode='probabilistic')
        # Run EpiMax (Possibilistic) logic
        acc_poss = run_extrapolation(model, L, num_samples=SAMPLES, mode='possibilistic')
        
        print(f"{L:<15} | {acc_prob:>18.2f}% | {acc_poss:>18.2f}%")
        
        results["len"].append(L)
        results["prob"].append(acc_prob)
        results["poss"].append(acc_poss)

    print("\n[Scientific Summary]")
    # Calculate the average decay rate
    prob_decay = results["prob"][0] - results["prob"][-1]
    poss_decay = results["poss"][0] - results["poss"][-1]
    print(f"Probabilistic Total Decay (L=2 to 20): {prob_decay:.2f}%")
    print(f"Possibilistic Total Decay (L=2 to 20): {poss_decay:.2f}%")

    print("\nGenerating epistemic plots for L=15...")
    plot_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=15, shuffle=True, drop_last=True)
    
    # We pass the pre-trained model and the new dataloader
    generate_epistemic_plots(model, plot_loader, chain_length=15)


if __name__ == "__main__":
    main()