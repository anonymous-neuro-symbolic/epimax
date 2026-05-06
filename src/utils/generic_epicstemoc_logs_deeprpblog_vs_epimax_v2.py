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

def generate_epistemic_plots(model, test_loader, chain_length=15, tau=100.0):
    model.eval()
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Define configurations for EpiMax
    epi_configs = {
        'epi_base':    {'alpha': 1.0, 'beta': 1.0, 'label': r'EpiMax (Base: $\alpha$=1, $\beta$=1)', 'color': '#ff7f0e'},
        'epi_strict':  {'alpha': 1.0, 'beta': 2.0, 'label': r'EpiMax (Strict: $\alpha$=1, $\beta$=2)', 'color': '#d62728'},
        'epi_relaxed': {'alpha': 2.0, 'beta': 0.5, 'label': r'EpiMax (Relaxed: $\alpha$=2, $\beta$=0.5)', 'color': '#9467bd'}
    }

    data = {
        'prob': {'confs': [], 'correct': []},
        'map':  {'confs': [], 'correct': []},
        'ltn':  {'confs': [], 'correct': []},
        **{k: {'confs': [], 'correct': []} for k in epi_configs.keys()}
    }
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
            logits = model(images)
            target_sum = labels.sum().item()
            probs = torch.softmax(logits, dim=1)
            outputs_prob = [probs[j:j+1] for j in range(chain_length)]
            
            # 1. DeepProbLog & MAP & LTN
            for mode, func in [('prob', probabilistic_addition), ('map', _map_addition), ('ltn', _ltn_addition)]:
                dist = func(outputs_prob) if mode == 'prob' else func(outputs_prob)
                pred_idx = dist.argmax(dim=1)[0]
                data[mode]['confs'].append(dist[0, pred_idx].item())
                data[mode]['correct'].append(1 if pred_idx.item() == target_sum else 0)
            
            # 2. EpiMax Variants
            for key, cfg in epi_configs.items():
                alpha, beta = cfg['alpha'], cfg['beta']
                z_max, _ = torch.max(logits, dim=1, keepdim=True)
                # Apply alpha to possibility mapping
                outputs_pi = [torch.exp(alpha * (logits[j:j+1] - z_max[j:j+1])) for j in range(chain_length)]
                dist_pi = possibilistic_addition(outputs_pi, tau=tau)
                
                pred_idx = dist_pi.argmax(dim=1)[0]
                mask = torch.ones_like(dist_pi, dtype=torch.bool)
                mask[0, pred_idx] = False
                
                # Apply beta to Necessity calculation
                max_comp_pi = dist_pi.masked_fill(~mask, 0.0).max().item()
                predicted_N = 1.0 - (max_comp_pi ** beta)
                
                data[key]['confs'].append(predicted_N)
                data[key]['correct'].append(1 if pred_idx.item() == target_sum else 0)

    # Convert to arrays
    for k in data:
        data[k]['confs'] = np.array(data[k]['confs'])
        data[k]['correct'] = np.array(data[k]['correct'])

    sns.set_theme(style="whitegrid")
    colors = {'prob': '#1f77b4', 'map': '#2ca02c', 'ltn': '#7f7f7f', **{k: v['color'] for k, v in epi_configs.items()}}
    labels = {'prob': 'DeepProbLog', 'map': 'MAP Inference', 'ltn': 'LTN', **{k: v['label'] for k, v in epi_configs.items()}}

    # --- PLOT 1: Calibration Curve ---
    plt.figure(figsize=(9, 8))
    bins = np.linspace(0, 1.0, 10)
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect calibration", alpha=0.5)
    
    for k in data.keys():
        indices = np.digitize(data[k]['confs'], bins) - 1
        bin_accs, bin_confs = [], []
        for i in range(len(bins)-1):
            mask = indices == i
            if np.sum(mask) > 0:
                bin_accs.append(np.mean(data[k]['correct'][mask]))
                bin_confs.append(np.mean(data[k]['confs'][mask]))
        
        # Use lines with dots for all, specifically for EpiMax variants
        plt.plot(bin_confs, bin_accs, 'o-', color=colors[k], label=labels[k], 
                 linewidth=2.5, markersize=8, alpha=0.8)
    
    plt.title(f"Calibration Sensitivity Analysis (L={chain_length})", fontsize=24)
    plt.xlabel("Confidence / Necessity", fontsize=22)
    plt.ylabel("Accuracy", fontsize=22)
    plt.legend(loc='lower right', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"calibration_sensitivity_L{chain_length}.png"), dpi=300)

    # --- PLOT 2: Epistemic State Distribution ---
    plt.figure(figsize=(12, 7))
    groups = ['Correct predictions', 'Incorrect predictions']
    x = np.arange(len(groups))
    width = 0.12
    all_keys = list(data.keys())
    
    for idx, k in enumerate(all_keys):
        cor_mask, inc_mask = data[k]['correct'] == 1, data[k]['correct'] == 0
        total = len(data[k]['correct'])
        offset = (idx - (len(all_keys)/2)) * width + (width/2)
        
        c_cert = np.sum(data[k]['confs'][cor_mask] >= 0.5) / total * 100
        c_unc  = np.sum(data[k]['confs'][cor_mask] < 0.5) / total * 100
        i_cert = np.sum(data[k]['confs'][inc_mask] >= 0.5) / total * 100
        i_unc  = np.sum(data[k]['confs'][inc_mask] < 0.5) / total * 100
        
        plt.bar(x + offset, [c_cert, i_cert], width, color=colors[k], alpha=0.9, label=labels[k])
        plt.bar(x + offset, [c_unc, i_unc], width, bottom=[c_cert, i_cert], color=colors[k], alpha=0.3, hatch='//')

    plt.title(f"Epistemic Regulation Breakdown (L={chain_length})", fontsize=24)
    plt.ylabel("Total Predictions (%)", fontsize=22)
    plt.xticks(x, groups, fontsize=22)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"epistemic_breakdown_sensitivity_L{chain_length}.png"), dpi=300)
    plt.close()