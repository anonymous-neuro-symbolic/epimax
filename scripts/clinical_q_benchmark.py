import torch
import torch.optim as optim
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from src.datasets.clinical_loader import get_clinical_ood_split
from src.models.tabular_centroid_network import TabularCentroidNet
from src.losses.joint_q_losses import JointQSupremumLoss, q_exp

def calculate_metrics(id_uncert, ood_uncert):
    """
    Standard OOD metric calculation using uncertainty scores.
    """
    y_true = np.concatenate([np.zeros(len(id_uncert)), np.ones(len(ood_uncert))])
    y_scores = np.concatenate([id_uncert, ood_uncert])
    
    auroc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    # Find FPR at 95% TPR
    idx = np.where(tpr >= 0.95)[0][0]
    fpr95 = fpr[idx]
    
    return {"auroc": auroc, "aupr": aupr, "fpr95": fpr95}

def plot_uncertainty_density(id_scores, ood_scores, save_path):
    """
    Visualizes the epistemic manifold separation between age groups.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(id_scores, bins=50, alpha=0.6, density=True, label='ID (Adults 18-60)', color='blue')
    plt.hist(ood_scores, bins=50, alpha=0.6, density=True, label='OOD (Elderly 80+)', color='orange')
    plt.title('Clinical epistemic uncertainty: Joint $q$-supremum', fontsize=14)
    plt.xlabel(r'Epistemic uncertainty $U = 1 - \max \pi$', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(loc='upper left')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def run_clinical_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = "results/clinical_benchmark"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Loading clinical data (Age-based covariate shift)
    # ID: Adults (18-60), OOD: Elderly (80+)
    train_loader, ood_loader = get_clinical_ood_split(batch_size=64)
    
    # 2. Initializing SN-MLP and Joint q-Loss
    # input_dim=3 (heart_rate, sys_bp, age)
    model = TabularCentroidNet(input_dim=3, hidden_dim=128, latent_dim=64, num_classes=2).to(device)
    
    q_val = 0.5
    criterion = JointQSupremumLoss(q=q_val, lambda_n=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 3. Training phase
    print(">>> Starting clinical training on adults (18-60)...")
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch}/{num_epochs-1}] - Avg loss: {running_loss/len(train_loader):.4f}")

    # 4. Evaluation phase
    model.eval()
    def extract_uncertainties(loader):
        scores = []
        with torch.no_grad():
            for features, _ in loader:
                logits = model(features.to(device))
                # Map logits to possibilities using the Tsallis q-exponential
                pi = q_exp(logits, q=q_val)
                # U = 1 - max(pi)
                uncertainty = 1.0 - torch.max(pi, dim=1)[0]
                scores.append(uncertainty.cpu().numpy())
        return np.concatenate(scores)

    print(">>> Extracting uncertainty for ID and OOD clinical populations...")
    id_uncert = extract_uncertainties(train_loader)
    ood_uncert = extract_uncertainties(ood_loader)
    
    # 5. Metrics and visualization
    metrics = calculate_metrics(id_uncert, ood_uncert)
    
    print("\n=== Clinical Benchmark Results (Age Shift) ===")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"AUPR:  {metrics['aupr']:.4f}")
    print(f"FPR95: {metrics['fpr95']:.4f}")

    # Save artifacts
    with open(f"{results_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    plot_uncertainty_density(
        id_uncert, 
        ood_uncert, 
        f"{results_dir}/q_supremum_clinical_density.pdf"
    )
    
    print(f"\nResults saved to {results_dir}/")

if __name__ == "__main__":
    run_clinical_benchmark()