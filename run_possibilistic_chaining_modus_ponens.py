import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Balanced data loading strategy ---
def get_balanced_chain(dataset, L, is_critical=True):
    """Constructs a chain of L images that are either all >= 5 or have at least one < 5."""
    indices_ge_5 = np.where(dataset.targets >= 5)[0]
    indices_lt_5 = np.where(dataset.targets < 5)[0]
    
    chain_indices = []
    if is_critical:
        # All digits must be >= 5
        chain_indices = np.random.choice(indices_ge_5, L, replace=False)
    else:
        # At least one digit must be < 5
        num_lt_5 = np.random.randint(1, L + 1)
        num_ge_5 = L - num_lt_5
        idx_lt = np.random.choice(indices_lt_5, num_lt_5, replace=False)
        idx_ge = np.random.choice(indices_ge_5, num_ge_5, replace=False)
        chain_indices = np.concatenate([idx_lt, idx_ge])
        np.random.shuffle(chain_indices)
        
    images = torch.stack([dataset[i][0] for i in chain_indices])
    labels = torch.tensor([dataset[i][1] for i in chain_indices])
    return images, labels

# --- 2. Logic operators (Product vs Lukasiewicz vs Gödel) ---
def probabilistic_conjunction(probs):
    return torch.prod(probs)

def fuzzy_conjunction(probs):
    L = len(probs)
    return torch.clamp(torch.sum(probs) - (L - 1), min=0.0)

def possibilistic_conjunction(necessities):
    return torch.min(necessities)

def calculate_necessity(logits, alpha=1.0, beta=1.0):
    probs = torch.softmax(logits, dim=1)
    prob_ge_5 = probs[:, 5:].sum(dim=1)
    
    z_max, _ = torch.max(logits, dim=1, keepdim=True)
    pi = torch.exp((logits - z_max) / alpha)
    pi_competitors, _ = pi[:, :5].max(dim=1)
    N = torch.clamp(1.0 - (beta * pi_competitors), min=0.0, max=1.0)
    return N, prob_ge_5

# --- 3. PerceptionNet (Reuse your existing structure) ---
from src.models.perception_net import PerceptionNet

def main():
    # Load your pre-trained model
    model = PerceptionNet().to(DEVICE)
    # model.load_state_dict(torch.load('path_to_weights.pt')) 
    model.eval()

    test_dataset = datasets.MNIST('./data', train=False, download=True, 
                                  transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

    lengths = [2, 5, 10, 15, 20]
    SAMPLES = 200 # Total chains per length

    print(f"\nBalanced fault tree test (Oracle Accuracy target: 50%)")
    print(f"{'L':<5} | {'Oracle Acc':<12} | {'Prob Conf':<12} | {'Fuzzy Conf':<12} | {'Poss Conf':<12}")
    print("-" * 65)

    for L in lengths:
        metrics = {'prob': [], 'fuzzy': [], 'poss': [], 'oracle': []}
        
        for i in range(SAMPLES):
            # 50/50 split of Critical vs Non-Critical chains
            is_critical = (i % 2 == 0)
            images, labels = get_balanced_chain(test_dataset, L, is_critical=is_critical)
            images = images.to(DEVICE)
            
            with torch.no_grad():
                logits = model(images)
                necessities, probs_ge_5 = calculate_necessity(logits)
                
                # We only care about the confidence of the "True" class (Critical state)
                conf_prob = probabilistic_conjunction(probs_ge_5).item()
                conf_fuzzy = fuzzy_conjunction(probs_ge_5).item()
                conf_poss = possibilistic_conjunction(necessities).item()
                
                # Metrics for samples that are ACTUALLY critical
                if is_critical:
                    metrics['prob'].append(conf_prob)
                    metrics['fuzzy'].append(conf_fuzzy)
                    metrics['poss'].append(conf_poss)
                
                metrics['oracle'].append(1 if is_critical else 0)

        print(f"{L:<5} | {np.mean(metrics['oracle']):>10.1%} | {np.mean(metrics['prob']):>11.4f} | {np.mean(metrics['fuzzy']):>11.4f} | {np.mean(metrics['poss']):>11.4f}")

if __name__ == "__main__":
    main()