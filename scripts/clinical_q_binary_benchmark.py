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
from src.losses.joint_q_losses import JointQBinaryLoss, q_exp, q_log

def extract_bipolar_metrics(model, loader, device, q=0.5):
    """
    Extracts Possibility (pi) and Necessity (N) values.
    Note: For the independent case, Necessity is defined by the 
    most restrictive complement: N_j = 1 - max_{i!=j} pi_i.
    """
    model.eval()
    all_pi = []
    all_N = []
    
    with torch.no_grad():
        for features, _ in loader:
            logits = model(features.to(device)) 
            pi = q_exp(logits, q)
            
            # Global necessity definition for cross-experiment consistency
            # N_j = max(0, pi_j - max_{i!=j} pi_i)
            pi_max_others, _ = torch.max(
                pi.unsqueeze(1) * (1 - torch.eye(pi.size(1), device=device)).unsqueeze(0), 
                dim=2
            )
            necessity = torch.clamp(pi - pi_max_others, min=0.0)
            
            all_pi.append(pi.cpu().numpy())
            all_N.append(necessity.cpu().numpy())
            
    return np.concatenate(all_pi), np.concatenate(all_N)

def classify_epistemic_states(pi, N, threshold=0.5):
    """ Categorizes samples based on Dubois-Prade epistemic states. """
    max_pi = np.max(pi, axis=1)
    max_N = np.max(N, axis=1)
    
    ignorance = max_pi < threshold
    conflict = (max_pi >= threshold) & (max_N < threshold)
    certainty = max_N >= threshold
    
    return certainty, conflict, ignorance

def run_clinical_binary_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = "results/clinical_binary_comparison"
    os.makedirs(results_dir, exist_ok=True)
    
    # Using independent q-Binary loss with a margin m=2.0
    train_loader, ood_loader = get_clinical_ood_split(batch_size=64)
    model = TabularCentroidNet(input_dim=3, num_classes=2).to(device)
    q_val = 0.5
    criterion = JointQBinaryLoss(q=q_val, lambda_n=1.0, margin=0.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(">>> Optimizing independent q-binary manifold...")
    for epoch in range(50):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(features.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()

    # Bipolar Extraction
    pi_id, n_id = extract_bipolar_metrics(model, train_loader, device, q_val)
    pi_ood, n_ood = extract_bipolar_metrics(model, ood_loader, device, q_val)
    
    # Metrics
    u_id = 1.0 - np.max(pi_id, axis=1)
    u_ood = 1.0 - np.max(pi_ood, axis=1)
    y_true = np.concatenate([np.zeros(len(u_id)), np.ones(len(u_ood))])
    y_scores = np.concatenate([u_id, u_ood])
    auroc = roc_auc_score(y_true, y_scores)

    # Epistemic States
    id_states = classify_epistemic_states(pi_id, n_id)
    ood_states = classify_epistemic_states(pi_ood, n_ood)
    
    def get_percentages(states):
        total = len(states[0])
        return [f"{(np.sum(s)/total)*100:.2f}%" for s in states]

    table_data = [
        ["In-distribution (Adults)", *get_percentages(id_states)],
        ["Out-of-distribution (Elderly)", *get_percentages(ood_states)]
    ]
    
    print(f"\n=== Clinical Binary Benchmark Results (AUROC: {auroc:.4f}) ===")
    print(tabulate(table_data, headers=["Population", "Certainty", "Conflict", "Ignorance"], tablefmt="grid"))
    
    # Save results
    with open(f"{results_dir}/metrics.json", "w") as f:
        json.dump({"auroc": auroc}, f, indent=4)

if __name__ == "__main__":
    run_clinical_binary_benchmark()