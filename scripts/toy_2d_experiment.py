import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import os

from src.losses.joint_q_losses import JointQSupremumLoss, JointQBinaryLoss, q_exp

# ==========================================
# 1. DATASET GENERATION (ID and OOD)
# ==========================================
def get_datasets():
    centers = [[-2, -2], [2, -2], [0, 2]]
    # 1. In-Distribution (ID) Data
    X_id, y_id = make_blobs(n_samples=800, centers=centers, cluster_std=0.6, random_state=42)
    
    # Split into Train and Test
    X_train, y_train = X_id[:600], y_id[:600]
    X_test_id, _ = X_id[600:], y_id[600:]
    
    # 2. Out-of-Distribution (OOD) Data
    # Sample uniformly from the background [-6, 6], excluding the known clusters
    np.random.seed(42)
    X_ood_candidates = np.random.uniform(-6, 6, (3000, 2))
    X_test_ood = []
    
    for x in X_ood_candidates:
        # If distance to the closest center is > 2.5, it is safely OOD
        min_dist = min([np.linalg.norm(x - np.array(c)) for c in centers])
        if min_dist > 2.5:
            X_test_ood.append(x)
        if len(X_test_ood) == 200: # We need 200 OOD samples to match ID test size
            break
            
    X_test_ood = np.array(X_test_ood)
    
    return (torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long),
            torch.tensor(X_test_id, dtype=torch.float32), torch.tensor(X_test_ood, dtype=torch.float32))

# ==========================================
# 2. ARCHITECTURES
# ==========================================
class ToyMLP(nn.Module):
    """Standard architecture for Softmax baseline"""
    def __init__(self, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class ToyRBF(nn.Module):
    """Distance-to-Centroid architecture for Possibilistic Methods"""
    def __init__(self, num_classes=3, feature_dim=16):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, feature_dim)
        )
        self.centroids = nn.Parameter(torch.randn(num_classes, feature_dim))
        
    def forward(self, x):
        features = self.extractor(x)
        dist = torch.cdist(features, self.centroids, p=2) ** 2
        return -dist

# ==========================================
# 3. METRIC CALCULATION
# ==========================================
def calculate_ood_metrics(id_uncertainty, ood_uncertainty):
    # Higher uncertainty means more likely to be OOD (Positive Class = 1)
    y_true = np.concatenate([np.zeros(len(id_uncertainty)), np.ones(len(ood_uncertainty))])
    y_scores = np.concatenate([id_uncertainty, ood_uncertainty])
    
    auroc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    # Find FPR at 95% TPR
    idx = np.where(tpr >= 0.95)[0][0]
    fpr95 = fpr[idx]
    
    return auroc, aupr, fpr95

# ==========================================
# 4. TRAINING & EVALUATION ORCHESTRATOR
# ==========================================
def train_evaluate_plot(method_name, model, criterion, q_value, is_probabilistic=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    X_train, y_train, X_id_test, X_ood_test = get_datasets()
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_id_test, X_ood_test = X_id_test.to(device), X_ood_test.to(device)

    # 1. Training Loop
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    # 2. Evaluation (Quantitative Metrics)
    model.eval()
    with torch.no_grad():
        logits_id = model(X_id_test)
        logits_ood = model(X_ood_test)
        
        if is_probabilistic:
            # For Softmax: Uncertainty = 1 - max(softmax(logits))
            prob_id = torch.softmax(logits_id, dim=1)
            prob_ood = torch.softmax(logits_ood, dim=1)
            uncert_id = 1.0 - torch.max(prob_id, dim=1)[0]
            uncert_ood = 1.0 - torch.max(prob_ood, dim=1)[0]
        else:
            # For Q-Losses: Uncertainty = 1 - max(q_exp(logits))
            pi_id = q_exp(logits_id, q=q_value)
            pi_ood = q_exp(logits_ood, q=q_value)
            uncert_id = 1.0 - torch.max(pi_id, dim=1)[0]
            uncert_ood = 1.0 - torch.max(pi_ood, dim=1)[0]
            
    uncert_id = uncert_id.cpu().numpy()
    uncert_ood = uncert_ood.cpu().numpy()
    
    auroc, aupr, fpr95 = calculate_ood_metrics(uncert_id, uncert_ood)

    # 3. Visualization (Manifolds)
    xx, yy = np.meshgrid(np.linspace(-6, 6, 100), np.linspace(-6, 6, 100))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits_grid = model(grid)
        if is_probabilistic:
            prob_grid = torch.softmax(logits_grid, dim=1)
            uncertainty = 1.0 - torch.max(prob_grid, dim=1)[0]
        else:
            pi_grid = q_exp(logits_grid, q=q_value)
            uncertainty = 1.0 - torch.max(pi_grid, dim=1)[0]

    uncertainty = uncertainty.cpu().numpy().reshape(xx.shape)
    X_cpu, y_cpu = X_train.cpu().numpy(), y_train.cpu().numpy()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(xx, yy, uncertainty, levels=20, cmap='inferno', vmin=0.0, vmax=1.0)
    plt.colorbar(contour, label=r'Epistemic Uncertainty $U = 1 - \max \pi$')
    
    colors = ['cyan', 'magenta', 'lime']
    for class_idx in range(3):
        idx = y_cpu == class_idx
        plt.scatter(X_cpu[idx, 0], X_cpu[idx, 1], c=colors[class_idx], 
                    edgecolor='k', marker='o', s=30, label=f'Hypothesis $\pi_{class_idx+1}$')

    # NeuralIPS formatted plot titles
    if "Supremum" in method_name:
        title = r'Epistemic Manifold: Joint $q$-Supremum'
    elif "Binary" in method_name:
        title = r'Epistemic Manifold: Joint Independent $q$-Binary'
    else:
        title = r'Epistemic Manifold: Probabilistic Softmax'

    plt.title(title, fontsize=14)
    plt.xlabel(r'Spatial Coordinate $x_1$', fontsize=12)
    plt.ylabel(r'Spatial Coordinate $x_2$', fontsize=12)
    plt.legend(loc='upper right')
    
    os.makedirs("results/toy_experiments", exist_ok=True)
    plt.savefig(f"results/toy_experiments/{method_name}_manifold.pdf", bbox_inches='tight')
    plt.close()
    
    return {"Method": method_name, "AUROC": auroc, "AUPR": aupr, "FPR95": fpr95}

if __name__ == "__main__":
    q_val = 0.5
    results = []
    
    print("Running Spatial OOD Benchmarks...")
    
    # 1. Standard Softmax Baseline
    res = train_evaluate_plot("Baseline_Softmax", ToyMLP(num_classes=3), nn.CrossEntropyLoss(), q_val, is_probabilistic=True)
    results.append(res)
    
    # 2. Joint q-Supremum
    res = train_evaluate_plot("Ours_q-Supremum", ToyRBF(num_classes=3), JointQSupremumLoss(q=q_val, lambda_n=1.0), q_val)
    results.append(res)
    
    # 3. Joint q-Binary
    res = train_evaluate_plot("Ours_q-Binary", ToyRBF(num_classes=3), JointQBinaryLoss(q=q_val, lambda_n=1.0, margin=2.0), q_val)
    results.append(res)

    # Print LaTeX / Markdown formatted Table
    print("\n" + "="*60)
    print(f"{'Method':<20} | {'AUROC (↑)':<10} | {'AUPR (↑)':<10} | {'FPR95 (↓)':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['Method']:<20} | {r['AUROC']:.4f}     | {r['AUPR']:.4f}     | {r['FPR95']:.4f}")
    print("="*60 + "\n")