from src.demo.epistemic_losses import _get_competitor_possibility, dual_possibilistic_loss, syllogistic_chain_loss
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# --- 1. SETUP KINEMATIC PHASE SPACE DATA ---
def generate_phase_space_data(n_samples=4000):
    X = torch.empty(n_samples, 2).uniform_(-5.0, 5.0)
    A1 = (X[:, 0] < 0).float()  # Critical proximity
    A2 = (X[:, 0] >= 0).float() # Safe proximity
    A3 = (X[:, 1] < 0).float()  # Closing velocity
    A4 = (X[:, 1] >= 0).float() # Separating velocity
    B1 = (A1 * A3)
    B2 = (A2 * A3)
    B3 = (A1 * A4)
    B4 = (A2 * A4)
    return X, A1, A2, A3, A4, B1, B2, B3, B4

# --- 2. ARCHITECTURE ---
class PhaseSpacePerception(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.shared(x)

# --- 3. TRAINING LOOP ---
def train_comparative_models():
    X, A1, A2, A3, A4, B1, B2, B3, B4 = generate_phase_space_data(5000)
    
    # Model 1: Baseline for DeepProbLog & LTN (Standard BCE on the 4 premises)
    model_baseline = PhaseSpacePerception(4)
    # opt_base = optim.Adam(model_baseline.parameters(), lr=0.01)
    opt_epi = optim.Adam(model_epimax.parameters(), lr=0.005, weight_decay=0.05)
    criterion_bce = nn.BCEWithLogitsLoss()
    targets_base = torch.stack([A1, A2, A3, A4], dim=1)
    
    # Model 2: EpiMax (Training on premises and conclusions with logic constraints)
    # Using 8 heads (2 classes each: True/False) for the possibilistic logit formulation
    class EpiMaxModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
            self.heads = nn.ModuleList([nn.Linear(128, 2) for _ in range(8)])
        def forward(self, x):
            feat = self.shared(x)
            return [head(feat) for head in self.heads]
            
    model_epimax = EpiMaxModel()
    opt_epi = optim.Adam(model_epimax.parameters(), lr=0.005)
    targets_epi = [A1.long(), A2.long(), A3.long(), A4.long(), B1.long(), B2.long(), B3.long(), B4.long()]

    print("Training Baseline and EpiMax models...")
    for epoch in range(500):
        # Train Baseline (CE)
        opt_base.zero_grad()
        logits_base = model_baseline(X)
        loss_base = criterion_bce(logits_base, targets_base)
        loss_base.backward()
        opt_base.step()
        
        # Train EpiMax
        opt_epi.zero_grad()
        logits_epi = model_epimax(X)
        local_loss = sum([dual_possibilistic_loss(logits_epi[i], targets_epi[i]) for i in range(8)])
        rule1 = syllogistic_chain_loss([logits_epi[0], logits_epi[2]], [A1.long(), A3.long()], logits_epi[4], B1.long())
        rule2 = syllogistic_chain_loss([logits_epi[1], logits_epi[2]], [A2.long(), A3.long()], logits_epi[5], B2.long())
        rule3 = syllogistic_chain_loss([logits_epi[0], logits_epi[3]], [A1.long(), A4.long()], logits_epi[6], B3.long())
        rule4 = syllogistic_chain_loss([logits_epi[1], logits_epi[3]], [A2.long(), A4.long()], logits_epi[7], B4.long())
        loss_epi = local_loss + 5.0 * (rule1 + rule2 + rule3 + rule4)
        loss_epi.backward()
        opt_epi.step()

    return model_baseline, model_epimax

# --- 4. PLOTTING THE COMPARISON ---
def render_comparative_canopies(model_baseline, model_epimax):
    os.makedirs("results", exist_ok=True)
    grid_res = 100
    xx, yy = np.meshgrid(np.linspace(-5, 5, grid_res), np.linspace(-5, 5, grid_res))
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    
    with torch.no_grad():
        # Baseline Probabilities (Sigmoid)
        probs = torch.sigmoid(model_baseline(grid_points)).numpy()
        P_A1, P_A2, P_A3, P_A4 = probs[:, 0], probs[:, 1], probs[:, 2], probs[:, 3]
        
        # DeepProbLog (Product)
        DP_B1 = (P_A1 * P_A3).reshape(grid_res, grid_res)
        DP_B2 = (P_A2 * P_A3).reshape(grid_res, grid_res)
        DP_B3 = (P_A1 * P_A4).reshape(grid_res, grid_res)
        DP_B4 = (P_A2 * P_A4).reshape(grid_res, grid_res)
        
        # LTN (Łukasiewicz: max(0, a+b-1))
        LTN_B1 = np.maximum(0, P_A1 + P_A3 - 1).reshape(grid_res, grid_res)
        LTN_B2 = np.maximum(0, P_A2 + P_A3 - 1).reshape(grid_res, grid_res)
        LTN_B3 = np.maximum(0, P_A1 + P_A4 - 1).reshape(grid_res, grid_res)
        LTN_B4 = np.maximum(0, P_A2 + P_A4 - 1).reshape(grid_res, grid_res)
        
        # EpiMax (Gödel / Min via trained loss)
        logits_epi = model_epimax(grid_points)
        target_true = torch.ones(grid_points.size(0), dtype=torch.long)
        EP_B1 = (1.0 - _get_competitor_possibility(logits_epi[4], target_true)).numpy().reshape(grid_res, grid_res)
        EP_B2 = (1.0 - _get_competitor_possibility(logits_epi[5], target_true)).numpy().reshape(grid_res, grid_res)
        EP_B3 = (1.0 - _get_competitor_possibility(logits_epi[6], target_true)).numpy().reshape(grid_res, grid_res)
        EP_B4 = (1.0 - _get_competitor_possibility(logits_epi[7], target_true)).numpy().reshape(grid_res, grid_res)

    # Rendering the 1x3 Subplot
    fig = plt.figure(figsize=(24, 7))
    
    def plot_canopies(ax, B1, B2, B3, B4, title):
        ax.plot_surface(xx, yy, B1, color='red', alpha=0.8)
        ax.plot_surface(xx, yy, B2, color='orange', alpha=0.8)
        ax.plot_surface(xx, yy, B3, color='blue', alpha=0.8)
        ax.plot_surface(xx, yy, B4, color='green', alpha=0.8)
        ax.set_title(title, fontsize=18)
        ax.set_zlim(0, 1)
        # LOWER THE ELEVATION from 20 to 5 to see the sagging/clipping
        ax.view_init(elev=5, azim=-45) 
        ax.set_xlabel("Relative Distance")
        ax.set_ylabel("Relative Velocity")
        ax.set_zlabel("Confidence / Necessity")
    # def plot_canopies(ax, B1, B2, B3, B4, title):
    #     ax.plot_surface(xx, yy, B1, color='red', alpha=0.8)
    #     ax.plot_surface(xx, yy, B2, color='orange', alpha=0.8)
    #     ax.plot_surface(xx, yy, B3, color='blue', alpha=0.8)
    #     ax.plot_surface(xx, yy, B4, color='green', alpha=0.8)
    #     ax.set_title(title, fontsize=18)
    #     ax.set_zlim(0, 1)
    #     ax.view_init(elev=20, azim=-60)
    #     ax.set_xlabel("Relative Distance")
    #     ax.set_ylabel("Relative Velocity")
    #     ax.set_zlabel("Confidence / Necessity")

    ax1 = fig.add_subplot(131, projection='3d')
    plot_canopies(ax1, DP_B1, DP_B2, DP_B3, DP_B4, "DeepProbLog (Product t-norm)\nOverlapping Hallucinations")
    
    ax2 = fig.add_subplot(132, projection='3d')
    plot_canopies(ax2, LTN_B1, LTN_B2, LTN_B3, LTN_B4, "LTN (Łukasiewicz t-norm)\nUnnatural Geometrical Clipping")
    
    ax3 = fig.add_subplot(133, projection='3d')
    plot_canopies(ax3, EP_B1, EP_B2, EP_B3, EP_B4, "EpiMax (Possibilistic Topology)\nStrict, Calibrated Exclusivity")

    plt.tight_layout()
    plt.savefig("results/comparative_phase_space.png", dpi=300)
    print("Comparative visualization saved.")

if __name__ == "__main__":
    m_base, m_epi = train_comparative_models()
    render_comparative_canopies(m_base, m_epi)