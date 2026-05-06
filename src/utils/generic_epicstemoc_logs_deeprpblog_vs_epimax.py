import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.engine.logic_engine import possibilistic_addition, probabilistic_addition

def generate_epistemic_plots(model, test_loader, chain_length=15, tau=100.0):
    model.eval()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, '..', '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    prob_confidences = []
    epi_necessities = []
    correct_flags = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
            logits = model(images)
            target_sum = labels.sum().item()
            
            # --- DeepProbLog Branch ---
            probs = torch.softmax(logits, dim=1)
            outputs_prob = [probs[j:j+1] for j in range(chain_length)]
            final_dist_prob = probabilistic_addition(outputs_prob)
            pred_idx_prob = final_dist_prob.argmax(dim=1)[0]
            prob_confidences.append(final_dist_prob[0, pred_idx_prob].item())
            
            # --- EpiMax Branch ---
            z_max, _ = torch.max(logits, dim=1, keepdim=True)
            pi = torch.exp(logits - z_max)
            outputs_pi = [pi[j:j+1] for j in range(chain_length)]
            final_dist_pi = possibilistic_addition(outputs_pi, tau=tau)
            
            pred_idx_pi = final_dist_pi.argmax(dim=1)[0]
            
            mask = torch.ones_like(final_dist_pi, dtype=torch.bool)
            mask[0, pred_idx_pi] = False
            competitors = final_dist_pi.masked_fill(~mask, 0.0)
            predicted_N = 1.0 - competitors.max().item()
            
            epi_necessities.append(predicted_N)
            correct_flags.append(1 if pred_idx_pi.item() == target_sum else 0)

    # Convert to numpy arrays
    prob_confidences = np.array(prob_confidences)
    epi_necessities = np.array(epi_necessities)
    correct_flags = np.array(correct_flags)

    sns.set_theme(style="whitegrid")

    # ---------------------------------------------------------
    # PLOT 1: Reliability Diagram (Calibration Curve)
    # ---------------------------------------------------------
    plt.figure(figsize=(7, 6))
    bins = np.linspace(0, 1.0, 10)
    
    def calc_calibration(confidences, correct):
        bin_indices = np.digitize(confidences, bins) - 1
        bin_accs = []
        bin_confs = []
        for i in range(len(bins)-1):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_accs.append(np.mean(correct[mask]))
                bin_confs.append(np.mean(confidences[mask]))
        return bin_confs, bin_accs

    prob_confs, prob_accs = calc_calibration(prob_confidences, correct_flags)
    epi_confs, epi_accs = calc_calibration(epi_necessities, correct_flags)

    plt.plot([0, 1], [0, 1], 'k--', label="Perfect calibration", alpha=0.7)
    plt.plot(prob_confs, prob_accs, 'o-', color='blue', label="DeepProbLog (Probability)", linewidth=2, markersize=8)
    plt.plot(epi_confs, epi_accs, 's-', color='orange', label="EpiMax (Necessity)", linewidth=2, markersize=8)
    
    plt.title(f"Model calibration on logical chains (L={chain_length})", fontsize=14)
    plt.xlabel("Predicted confidence", fontsize=12)
    plt.ylabel("Empirical accuracy", fontsize=12)
    plt.legend(loc='lower right')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"calibration_curve_L{chain_length}.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 2: Epistemic State Distribution (Stacked Bar)
    # ---------------------------------------------------------
    # Define threshold for Certainty vs Conflict/Ignorance
    N_threshold = 0.5 
    
    correct_necessities = epi_necessities[correct_flags == 1]
    incorrect_necessities = epi_necessities[correct_flags == 0]
    
    states = {
        'Correct': [
            np.sum(correct_necessities >= N_threshold) / len(correct_flags) * 100, # Certain
            np.sum(correct_necessities < N_threshold) / len(correct_flags) * 100   # Uncertain (Conflict/Ignorance)
        ],
        'Incorrect': [
            np.sum(incorrect_necessities >= N_threshold) / len(correct_flags) * 100,
            np.sum(incorrect_necessities < N_threshold) / len(correct_flags) * 100
        ]
    }
    
    plt.figure(figsize=(7, 6))
    bar_width = 0.5
    
    p1 = plt.bar(['Correct predictions', 'Incorrect predictions'], 
                 [states['Correct'][0], states['Incorrect'][0]], 
                 bar_width, color='green', alpha=0.7, label='Certainty (N ≥ 0.5)')
    
    p2 = plt.bar(['Correct predictions', 'Incorrect predictions'], 
                 [states['Correct'][1], states['Incorrect'][1]], 
                 bar_width, bottom=[states['Correct'][0], states['Incorrect'][0]], 
                 color='red', alpha=0.7, label='Conflict / Ignorance (N < 0.5)')
                 
    plt.title(f"Epistemic state breakdown (L={chain_length})", fontsize=14)
    plt.ylabel("Percentage of total predictions (%)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"epistemic_breakdown_L{chain_length}.png"), dpi=300)
    plt.close()
    
    print(f"Alternative plots successfully generated in {results_dir}")