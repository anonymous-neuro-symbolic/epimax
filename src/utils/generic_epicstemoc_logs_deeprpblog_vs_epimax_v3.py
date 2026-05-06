import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.engine.logic_engine import possibilistic_addition, probabilistic_addition

def _ltn_addition(outputs):
    curr = outputs[0]
    for nxt in outputs[1:]:
        curr_len = curr.shape[1]
        nxt_len = nxt.shape[1]
        new_curr = torch.zeros((1, curr_len + nxt_len - 1), device=curr.device)
        for i in range(curr_len):
            for j in range(nxt_len):
                val = torch.clamp(curr[0, i] + nxt[0, j] - 1.0, min=0.0)
                new_curr[0, i+j] = torch.max(new_curr[0, i+j], val)
        curr = new_curr
    return curr

def _map_addition(outputs):
    curr = outputs[0]
    for nxt in outputs[1:]:
        curr_len = curr.shape[1]
        nxt_len = nxt.shape[1]
        new_curr = torch.zeros((1, curr_len + nxt_len - 1), device=curr.device)
        for i in range(curr_len):
            for j in range(nxt_len):
                val = curr[0, i] * nxt[0, j]
                new_curr[0, i+j] = torch.max(new_curr[0, i+j], val)
        curr = new_curr
    return curr / (curr.sum(dim=1, keepdim=True) + 1e-9)

def visualize_sample_logits(model, test_loader, device='cuda'):
    """Extracts and visualizes pre-activation logits for individual observations."""
    model.eval()
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract a single batch
    images, labels = next(iter(test_loader))
    images = images[:3].to(device)
    labels = labels[:3]
    
    with torch.no_grad():
        logits = model(images)
        
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    sns.set_theme(style="whitegrid")
    
    for i in range(3):
        # Image visualization
        img = images[i].cpu().squeeze().numpy()
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"Input perception (Ground truth: {labels[i].item()})", fontsize=20)
        axes[i, 0].axis('off')
        
        # Logit magnitude bar chart
        lgs = logits[i].cpu().numpy()
        axes[i, 1].bar(range(10), lgs, color='#1f77b4', alpha=0.8)
        axes[i, 1].set_title("Pre-activation logit magnitudes", fontsize=20)
        axes[i, 1].set_xticks(range(10))
        axes[i, 1].set_xlabel("Class index", fontsize=18)
        axes[i, 1].set_ylabel("Energy score (Logit)", fontsize=18)
        
        # Highlight the max logit
        max_idx = np.argmax(lgs)
        axes[i, 1].get_children()[max_idx].set_color('#d62728')
        
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "sample_logits_visualization.png"), dpi=300)
    plt.close()

def generate_accuracy_across_lengths(model, test_dataset, tau=100.0, device='cuda'):
    """Evaluates deductive accuracy across specified chain lengths and generates isolated plots."""
    model.eval()
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    chain_lengths = [2, 5, 10, 15, 20]
    num_samples = 500  # Number of chains to evaluate per length
    
    # Initialize metric storage
    metrics = {
        'prob': [], 'map': [], 'ltn': [],
        'epi_base': [], 'epi_strict': [], 'epi_relaxed': []
    }
    
    epi_configs = {
        'epi_base':    {'alpha': 1.0, 'beta': 1.0, 'label': r'EpiMax ($\alpha=1.0, \beta=1.0$)', 'color': '#ff7f0e'},
        'epi_strict':  {'alpha': 1.0, 'beta': 2.0, 'label': r'EpiMax ($\alpha=1.0, \beta=2.0$)', 'color': '#d62728'},
        'epi_relaxed': {'alpha': 2.0, 'beta': 0.5, 'label': r'EpiMax ($\alpha=2.0, \beta=0.5$)', 'color': '#9467bd'}
    }

    with torch.no_grad():
        for L in chain_lengths:
            print(f"Evaluating reasoning chain length L={L}...")
            # Use a dataloader with batch_size=L to easily group reasoning chains
            loader = torch.utils.data.DataLoader(test_dataset, batch_size=L, shuffle=True, drop_last=True)
            
            correct_counts = {k: 0 for k in metrics.keys()}
            total_evaluated = 0
            
            for i, (images, labels) in enumerate(loader):
                if i >= num_samples: break
                
                images = images.to(device)
                logits = model(images)
                target_sum = labels.sum().item()
                probs = torch.softmax(logits, dim=1)
                
                outputs_prob = [probs[j:j+1] for j in range(L)]
                
                # Baseline evaluations
                dist_prob = probabilistic_addition(outputs_prob)
                dist_map = _map_addition(outputs_prob)
                dist_ltn = _ltn_addition(outputs_prob)
                
                if dist_prob.argmax(dim=1)[0].item() == target_sum: correct_counts['prob'] += 1
                if dist_map.argmax(dim=1)[0].item() == target_sum: correct_counts['map'] += 1
                if dist_ltn.argmax(dim=1)[0].item() == target_sum: correct_counts['ltn'] += 1
                
                # EpiMax evaluations
                for key, cfg in epi_configs.items():
                    alpha = cfg['alpha']
                    z_max, _ = torch.max(logits, dim=1, keepdim=True)
                    outputs_pi = [torch.exp(alpha * (logits[j:j+1] - z_max[j:j+1])) for j in range(L)]
                    dist_pi = possibilistic_addition(outputs_pi, tau=tau)
                    
                    if dist_pi.argmax(dim=1)[0].item() == target_sum: correct_counts[key] += 1
                    
                total_evaluated += 1
                
            # Store percentage accuracies
            for k in metrics.keys():
                metrics[k].append(correct_counts[k] / total_evaluated)

    sns.set_theme(style="whitegrid")

    # --- Plot 1: Baseline probability accuracy ---
    plt.figure(figsize=(8, 6))
    plt.plot(chain_lengths, metrics['prob'], 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='DeepProbLog')
    plt.plot(chain_lengths, metrics['map'], 's-', color='#2ca02c', linewidth=2.5, markersize=8, label='MAP (Max-Product)')
    plt.plot(chain_lengths, metrics['ltn'], '^-', color='#7f7f7f', linewidth=2.5, markersize=8, label='LTN (Fuzzy \L{}ukasiewicz)')
    
    plt.title("Extrapolation accuracy of standard baselines", fontsize=16)
    plt.xlabel("Reasoning chain length ($L$)", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(0, 1.05)
    plt.xticks(chain_lengths)
    plt.legend(loc='lower left', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "accuracy_baselines_panel_A.png"), dpi=300)
    plt.close()

    # --- Plot 2: EpiMax variant accuracy ---
    plt.figure(figsize=(8, 6))
    for key, cfg in epi_configs.items():
        # Enforcing dotted lines for all EpiMax variants as requested
        plt.plot(chain_lengths, metrics[key], linestyle=':', marker='o', color=cfg['color'], 
                 linewidth=2.5, markersize=8, label=cfg['label'])
        
    plt.title("Extrapolation accuracy of possibilistic configurations", fontsize=16)
    plt.xlabel("Reasoning chain length ($L$)", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(0, 1.05)
    plt.xticks(chain_lengths)
    plt.legend(loc='lower left', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "accuracy_epimax_panel_C.png"), dpi=300)
    plt.close()

# Existing generate_epistemic_plots function remains below...
# (Omitted for brevity, but you retain your original calibration plotting function here)