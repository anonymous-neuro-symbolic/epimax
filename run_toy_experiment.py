import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time

from src.datasets.synthetic_2d import get_synthetic_dataloaders
from src.losses.epimax_loss import DualPossibilisticLoss, EpiMax
from src.losses.edl_loss import EDLLoss
from src.models.duq_model import DUQModel, duq_loss
from src.utils.plotter import plot_variance_heatmaps, plot_gamma_sensitivity

# 1. Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
K_ROUNDS = 10
EPOCHS = 100
LR = 0.01
SIGMA_DUQ = 1.0
RESOLUTION = 0.1

def get_mlp():
    return nn.Sequential(
        nn.Linear(2, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 3)
    ).to(DEVICE)

# 2. Setup
train_loader, _, dataset = get_synthetic_dataloaders(batch_size=64)
pad = 2.0
xx, yy = np.meshgrid(
    np.arange(dataset.x_min - pad, dataset.x_max + pad, RESOLUTION),
    np.arange(dataset.y_min - pad, dataset.y_max + pad, RESOLUTION)
)
grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(DEVICE)

results = {'epimax_n': [], 'epimax_pi': [], 'edl': [], 'duq': []}
overhead = {'epimax': [], 'edl': [], 'duq': []}

# 3. Main stability loop
for r in range(K_ROUNDS):
    print(f"--- Round {r+1}/{K_ROUNDS} ---")
    
    # Re-init models
    m_epi, m_edl, m_duq = get_mlp(), get_mlp(), DUQModel(sigma=SIGMA_DUQ).to(DEVICE)
    c_epi, c_edl = DualPossibilisticLoss(gamma=2.0), EDLLoss(num_classes=3)
    epimax_layer = EpiMax().to(DEVICE)
    
    opts = {
        'epimax': optim.Adam(m_epi.parameters(), lr=LR),
        'edl': optim.Adam(m_edl.parameters(), lr=LR),
        'duq': optim.Adam(m_duq.parameters(), lr=LR)
    }

    # Training with timing
    for name in ['epimax', 'edl', 'duq']:
        model = m_epi if name == 'epimax' else (m_edl if name == 'edl' else m_duq)
        optimizer = opts[name]
        model.train()
        
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        t0 = time.time()
        
        for epoch in range(EPOCHS):
            for X, y in train_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                if name == 'epimax': loss = c_epi(model(X), y)
                elif name == 'edl': loss = c_edl(model(X), y, epoch)
                else: loss = duq_loss(model(X), y)
                loss.backward(); optimizer.step()
        
        torch.cuda.synchronize() if DEVICE == 'cuda' else None
        overhead[name].append(time.time() - t0)

    # Evaluation
    m_epi.eval(); m_edl.eval(); m_duq.eval()
    with torch.no_grad():
        pi, N = epimax_layer(m_epi(grid_tensor))
        results['epimax_pi'].append(torch.max(pi, 1)[0].cpu().numpy())
        results['epimax_n'].append(torch.max(N, 1)[0].cpu().numpy())
        
        u_edl = 3.0 / torch.sum(F.relu(m_edl(grid_tensor)) + 1.0, 1)
        results['edl'].append(u_edl.cpu().numpy())
        
        u_duq = 1.0 - torch.max(m_duq(grid_tensor), 1)[0]
        results['duq'].append(u_duq.cpu().numpy())

# 4. Reporting
stability_data = []
for k, v in results.items():
    arr = np.stack(v)
    std_map = np.std(arr, axis=0)
    stability_data.append({
        "Metric": k.upper(),
        "Mean Std (↓)": np.mean(std_map),
        "Consistency (↑)": 1.0 - np.mean(std_map)
    })

print("\n" + "="*50)
print("STABILITY STATISTICS")
print("="*50)
print(pd.DataFrame(stability_data).to_string(index=False))

print("\n" + "="*50)
print("COMPUTATIONAL OVERHEAD")
print("="*50)
for k, v in overhead.items():
    print(f"{k.upper():<8}: {np.mean(v):.2f}s ± {np.std(v):.2f}s")

# 5. Plotting
print("\nSaving results...")
stats_maps = {k: np.std(np.stack(v), axis=0) for k, v in results.items()}
plot_variance_heatmaps(stats_maps, dataset, xx, yy, save_path="results/stability_heatmaps.png")
plot_gamma_sensitivity(save_path="results/gamma_curves.png")