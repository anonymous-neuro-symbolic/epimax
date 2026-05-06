import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from src.models.perception_net import PerceptionNet
from src.engine.logic_engine import possibilistic_addition, probabilistic_addition

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def pretrain_perception():
    model = PerceptionNet().to(DEVICE)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=256, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    print("Pre-training PerceptionNet...")
    for img, label in train_loader:
        img, label = img.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(img), label)
        loss.backward()
        optimizer.step()
    return model

def main():
    model = pretrain_perception()
    model.eval()
    
    chain_length = 15
    num_samples = 1000
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=chain_length, shuffle=True, drop_last=True)

    print(f"\nExtracting logic chains for L={chain_length}...")
    
    # Store predictions for DeepProbLog and 3 EpiMax configs
    results = {
        'DeepProbLog': {'conf': [], 'acc': []},
        'EpiMax (Baseline α=1, β=1)': {'conf': [], 'acc': []},
        'EpiMax (Strict α=1, β=2)': {'conf': [], 'acc': []},
        'EpiMax (Relaxed α=2, β=0.5)': {'conf': [], 'acc': []}
    }

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= num_samples: break
            images = images.to(DEVICE)
            target_sum = labels.sum().item()
            
            logits = model(images)
            
            # --- 1. DeepProbLog ---
            probs = torch.softmax(logits, dim=1)
            final_dist_prob = probabilistic_addition([probs[j:j+1] for j in range(chain_length)])
            pred_idx_prob = final_dist_prob.argmax(dim=1)[0].item()
            results['DeepProbLog']['conf'].append(final_dist_prob[0, pred_idx_prob].item())
            results['DeepProbLog']['acc'].append(1 if pred_idx_prob == target_sum else 0)
            
            # --- 2. EpiMax Sweep ---
            z_max, _ = torch.max(logits, dim=1, keepdim=True)
            
            configs = [
                ('EpiMax (Baseline α=1, β=1)', 1.0, 1.0),
                ('EpiMax (Strict α=1, β=2)', 1.0, 2.0),
                ('EpiMax (Relaxed α=2, β=0.5)', 2.0, 0.5)
            ]
            
            for label, alpha, beta in configs:
                # Apply alpha temperature
                pi = torch.exp((logits - z_max) / alpha)
                final_dist_pi = possibilistic_addition([pi[j:j+1] for j in range(chain_length)], tau=100.0)
                
                pred_idx_pi = final_dist_pi.argmax(dim=1)[0].item()
                
                # Calculate Necessity with beta penalty
                mask = torch.ones_like(final_dist_pi, dtype=torch.bool)
                mask[0, pred_idx_pi] = False
                competitors = final_dist_pi.masked_fill(~mask, 0.0)
                predicted_N = max(0.0, 1.0 - (beta * competitors.max().item()))
                
                results[label]['conf'].append(predicted_N)
                results[label]['acc'].append(1 if pred_idx_pi == target_sum else 0)

    # Convert to numpy arrays
    for key in results:
        results[key]['conf'] = np.array(results[key]['conf'])
        results[key]['acc'] = np.array(results[key]['acc'])

    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # 1. Calibration Shift Envelope
    plt.figure(figsize=(8, 6))
    bins = np.linspace(0, 1.0, 10)
    
    def get_cal_curve(conf, acc):
        bin_indices = np.digitize(conf, bins) - 1
        x, y = [], []
        for j in range(len(bins)-1):
            mask = bin_indices == j
            if np.sum(mask) > 0:
                x.append(np.mean(conf[mask]))
                y.append(np.mean(acc[mask]))
        return x, y

    plt.plot([0, 1], [0, 1], 'k--', label="Perfect calibration", alpha=0.7)
    
    colors = {'DeepProbLog': 'blue', 'EpiMax (Baseline α=1, β=1)': 'orange', 
              'EpiMax (Strict α=1, β=2)': 'red', 'EpiMax (Relaxed α=2, β=0.5)': 'green'}
    styles = {'DeepProbLog': '-', 'EpiMax (Baseline α=1, β=1)': '-', 
              'EpiMax (Strict α=1, β=2)': '--', 'EpiMax (Relaxed α=2, β=0.5)': ':'}

    for label in results:
        x, y = get_cal_curve(results[label]['conf'], results[label]['acc'])
        plt.plot(x, y, color=colors[label], linestyle=styles[label], marker='o', linewidth=2.5, label=label)

    plt.title(f"Calibration parameter shift (L={chain_length})", fontsize=18)
    plt.xlabel("Predicted confidence / Necessity", fontsize=18)
    plt.ylabel("Empirical accuracy", fontsize=18)
    plt.legend(fontsize=18)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "sensitivity_calibration_L15.png"), dpi=300)
    plt.close()

    # 2. Epistemic State Shift (Focusing on Incorrect Predictions)
    plt.figure(figsize=(8, 6))
    
    # We want to see how many INCORRECT predictions are dangerously labeled "Certain"
    labels_list = list(results.keys())
    false_certainty_rates = []
    safe_uncertainty_rates = []
    
    for label in labels_list:
        conf = results[label]['conf']
        acc = results[label]['acc']
        
        incorrect_mask = (acc == 0)
        incorrect_confs = conf[incorrect_mask]
        
        total_incorrect = len(incorrect_confs)
        if total_incorrect == 0: total_incorrect = 1
        
        # N >= 0.5 (or Prob >= 0.5) is considered "Certain"
        f_certain = np.sum(incorrect_confs >= 0.5) / total_incorrect * 100
        s_uncertain = np.sum(incorrect_confs < 0.5) / total_incorrect * 100
        
        false_certainty_rates.append(f_certain)
        safe_uncertainty_rates.append(s_uncertain)

    x = np.arange(len(labels_list))
    width = 0.6
    
    plt.bar(x, false_certainty_rates, width, color='darkred', alpha=0.8, label='Danger: Confidently Wrong (N ≥ 0.5)')
    plt.bar(x, safe_uncertainty_rates, width, bottom=false_certainty_rates, color='lightcoral', alpha=0.8, label='Safe: Conflict/Ignorance (N < 0.5)')
    
    plt.title(f"State of incorrect predictions by parameter (L={chain_length})", fontsize=18)
    plt.ylabel("Percentage of incorrect predictions (%)", fontsize=18)
    plt.xticks(x, [l.replace(' ', '\n') for l in labels_list], rotation=0, fontsize=18)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "sensitivity_epistemic_L15.png"), dpi=300)
    plt.close()

    print(f"Sensitivity plots successfully generated in {results_dir}")

if __name__ == "__main__":
    main()