import torch
import torch.optim as optim
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from tabulate import tabulate

from src.datasets.clinical_loader import get_clinical_ood_split
from src.models.tabular_centroid_network import TabularCentroidNet
from src.losses.joint_q_losses import JointQSupremumLoss, q_exp, q_log

def extract_bipolar_metrics(model, loader, device, q=0.5):
    """
    Extracts explicit possibility (pi) and necessity (N) values 
    based on the joint q-supremum manifold logic.
    """
    model.eval()
    all_pi = []
    all_N = []
    
    with torch.no_grad():
        for features, _ in loader:
            logits = model(features.to(device)) 
            
            # 1. Calculate possibility (pi) 
            pi = q_exp(logits, q)
            
            # 2. Calculate necessity (N) 
            # N_j = 1 - log_q(sum of others)
            pi_sum = torch.sum(pi, dim=1, keepdim=True)
            pi_others = torch.clamp(pi_sum - pi, min=1e-8)
            
            # pi_y_c represents the possibility of the complement
            pi_y_c = torch.relu(q_log(pi_others, q))
            necessity = torch.clamp(1.0 - pi_y_c, min=0.0)
            
            all_pi.append(pi.cpu().numpy())
            all_N.append(necessity.cpu().numpy())
            
    return np.concatenate(all_pi), np.concatenate(all_N)

def classify_epistemic_states(pi, N, threshold=0.5):
    """
    Categorizes samples into certainty, conflict, or ignorance.
    """
    max_pi = np.max(pi, axis=1)
    max_N = np.max(N, axis=1)
    
    # Ignorance: No class is consistent with the data
    ignorance = max_pi < threshold
    
    # Conflict: Data is consistent with multiple classes 
    conflict = (max_pi >= threshold) & (max_N < threshold)
    
    # Certainty: One class is clearly dominant
    certainty = max_N >= threshold
    
    return certainty, conflict, ignorance

def plot_bipolar_behavior(pi, N, is_ood, save_path):
    """
    Visualization of the relationship between pi and N.
    """
    plt.figure(figsize=(12, 5))
    
    # Uncertainty metric U = 1 - max(pi)
    u_id = 1.0 - np.max(pi[~is_ood], axis=1)
    u_ood = 1.0 - np.max(pi[is_ood], axis=1)
    
    plt.subplot(1, 2, 1)
    plt.hist(u_id, bins=40, alpha=0.6, label='ID uncertainty', color='blue', density=True)
    plt.hist(u_ood, bins=40, alpha=0.6, label='OOD uncertainty', color='orange', density=True)
    plt.title('Possibilistic uncertainty ($1 - \max \pi$)')
    plt.xlabel('Uncertainty value')
    plt.ylabel('Density')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(np.max(pi[~is_ood], axis=1), np.max(N[~is_ood], axis=1), 
                alpha=0.2, label='ID samples', color='blue', s=5)
    plt.scatter(np.max(pi[is_ood], axis=1), np.max(N[is_ood], axis=1), 
                alpha=0.2, label='OOD samples', color='orange', s=5)
    plt.plot([0, 1], [0, 1], 'r--', label='N = pi boundary')
    plt.title('Bipolarity consistency check ($N \leq \pi$)')
    plt.xlabel('Possibility ($\pi$)')
    plt.ylabel('Necessity ($N$)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_clinical_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = "results/clinical_bipolar"
    os.makedirs(results_dir, exist_ok=True)
    
    train_loader, ood_loader = get_clinical_ood_split(batch_size=64)
    model = TabularCentroidNet(input_dim=3, num_classes=2).to(device)
    q_val = 0.5
    criterion = JointQSupremumLoss(q=q_val)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(">>> Optimizing joint q-supremum manifold...")
    for epoch in range(50):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(features.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()

    # Bipolar extraction
    pi_id, n_id = extract_bipolar_metrics(model, train_loader, device, q_val)
    pi_ood, n_ood = extract_bipolar_metrics(model, ood_loader, device, q_val)
    
    # Metric calculation
    u_id = 1.0 - np.max(pi_id, axis=1)
    u_ood = 1.0 - np.max(pi_ood, axis=1)
    y_true = np.concatenate([np.zeros(len(u_id)), np.ones(len(u_ood))])
    y_scores = np.concatenate([u_id, u_ood])
    auroc = roc_auc_score(y_true, y_scores)

    # Epistemic state classification
    id_states = classify_epistemic_states(pi_id, n_id)
    ood_states = classify_epistemic_states(pi_ood, n_ood)
    
    def get_percentages(states):
        total = len(states[0])
        return [f"{(np.sum(s)/total)*100:.2f}%" for s in states]

    table_data = [
        ["In-distribution (Adults)", *get_percentages(id_states)],
        ["Out-of-distribution (Elderly)", *get_percentages(ood_states)]
    ]
    
    print(f"\n=== Clinical benchmark results (AUROC: {auroc:.4f}) ===")
    print(tabulate(table_data, headers=["Population", "Certainty", "Conflict", "Ignorance"], tablefmt="grid"))
    
    plot_bipolar_behavior(
        np.concatenate([pi_id, pi_ood]), 
        np.concatenate([n_id, n_ood]), 
        y_true.astype(bool), 
        f"{results_dir}/bipolar_behavior.pdf"
    )
    print(f"\nResults and plots saved to {results_dir}/")

if __name__ == "__main__":
    run_clinical_benchmark()