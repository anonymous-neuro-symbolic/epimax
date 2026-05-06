import torch
import torch.optim as optim
import numpy as np
import os
from tabulate import tabulate

from src.datasets.clinical_loader import get_clinical_ood_split
from src.models.tabular_centroid_network import TabularCentroidNet
from src.losses.joint_q_losses import JointQSupremumLoss
from src.losses.outlier_penalties import active_repulsive_loss
from scripts.clinical_q_benchmark2 import extract_bipolar_metrics, classify_epistemic_states

def generate_background_noise(batch_size, input_dim=3, device='cpu'):
    """
    Generates generic physiological background noise (Outlier Exposure D_out).
    This simulates samples that are mathematically possible but semantically void.
    """
    return torch.randn(batch_size, input_dim).to(device)

def run_outlier_exposure_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = "results/clinical_oe_refinement"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Loading clinical data
    train_loader, ood_loader = get_clinical_ood_split(batch_size=64)
    model = TabularCentroidNet(input_dim=3, num_classes=2).to(device)
    
    q_val = 0.5
    lambda_oe = 0.5 # Hyperparameter for the repulsive force
    criterion_id = JointQSupremumLoss(q=q_val)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(">>> Phase 2: Training with active outlier exposure...")
    for epoch in range(50):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Generate background outliers for this batch
            bg_data = generate_background_noise(features.size(0), device=device)
            
            optimizer.zero_grad()
            
            # ID Loss: Standard joint q-supremum anchoring
            logits_id = model(features)
            loss_id = criterion_id(logits_id, labels)
            
            # OOD Loss: Active repulsive gradient
            logits_oe = model(bg_data)
            loss_oe = active_repulsive_loss(logits_oe, q=q_val)
            
            # Total Objective: Composite manifold refinement
            total_loss = loss_id + (lambda_oe * loss_oe)
            
            total_loss.backward()
            optimizer.step()

    # 2. Bipolar evaluation of the refined manifold
    pi_id, n_id = extract_bipolar_metrics(model, train_loader, device, q_val)
    pi_ood, n_ood = extract_bipolar_metrics(model, ood_loader, device, q_val)
    
    id_states = classify_epistemic_states(pi_id, n_id)
    ood_states = classify_epistemic_states(pi_ood, n_ood)
    
    def get_percentages(states):
        total = len(states[0])
        return [f"{(np.sum(s)/total)*100:.2f}%" for s in states]

    table_data = [
        ["In-distribution (Adults)", *get_percentages(id_states)],
        ["Out-of-distribution (Elderly)", *get_percentages(ood_states)]
    ]
    
    print("\n=== Refined clinical results (Approach B: Outlier exposure) ===")
    print(tabulate(table_data, headers=["Population", "Certainty", "Conflict", "Ignorance"], tablefmt="grid"))

if __name__ == "__main__":
    run_outlier_exposure_experiment()