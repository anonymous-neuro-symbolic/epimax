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
    
    # 4 Perceptual Premises
    A1 = (X[:, 0] < 0).long()  # Critical proximity
    A2 = (X[:, 0] >= 0).long() # Safe proximity
    A3 = (X[:, 1] < 0).long()  # Closing velocity
    A4 = (X[:, 1] >= 0).long() # Separating velocity
    
    # 4 Logical Conclusions (The Intersections)
    B1 = (A1 & A3).long() # Critical Brake
    B2 = (A2 & A3).long() # Prepare to Brake
    B3 = (A1 & A4).long() # Maintain Gap
    B4 = (A2 & A4).long() # Safe Cruise
    
    return X, A1, A2, A3, A4, B1, B2, B3, B4

# --- 2. EXPANDED ARCHITECTURE ---
class PhaseSpacePerception(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        # 8 Classification heads
        self.heads = nn.ModuleList([nn.Linear(128, 2) for _ in range(8)])
        
    def forward(self, x):
        feat = self.shared(x)
        return [head(feat) for head in self.heads]

# --- 3. TRAINING LOOP ---
def train_model():
    X, A1, A2, A3, A4, B1, B2, B3, B4 = generate_phase_space_data(5000)
    targets = [A1, A2, A3, A4, B1, B2, B3, B4]
    
    model = PhaseSpacePerception()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    print("Training 4-Rule Phase Space Logic...")
    for epoch in range(600):
        optimizer.zero_grad()
        logits_all = model(X)
        
        # Local Perception Loss (All 8 states)
        local_loss = sum([dual_possibilistic_loss(logits_all[i], targets[i]) for i in range(8)])
        
        # The 4 Syllogistic Rules
        rule1 = syllogistic_chain_loss([logits_all[0], logits_all[2]], [A1, A3], logits_all[4], B1)
        rule2 = syllogistic_chain_loss([logits_all[1], logits_all[2]], [A2, A3], logits_all[5], B2)
        rule3 = syllogistic_chain_loss([logits_all[0], logits_all[3]], [A1, A4], logits_all[6], B3)
        rule4 = syllogistic_chain_loss([logits_all[1], logits_all[3]], [A2, A4], logits_all[7], B4)
        
        logic_loss = rule1 + rule2 + rule3 + rule4
        total_loss = local_loss + 5.0 * logic_loss
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Local: {local_loss.item():.4f} | Logic: {logic_loss.item():.4f}")
            
    return model

# --- 4. PLOTTING THE FOUR CANOPIES ---
def render_phase_space(model):
    os.makedirs("results", exist_ok=True)
    grid_res = 120
    x_vals = np.linspace(-5, 5, grid_res)
    y_vals = np.linspace(-5, 5, grid_res)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    
    with torch.no_grad():
        logits_all = model(grid_points)
        target_true = torch.ones(grid_points.size(0), dtype=torch.long)
        
        # Extract necessities for the 4 conclusions
        N_B1 = (1.0 - _get_competitor_possibility(logits_all[4], target_true)).numpy().reshape(grid_res, grid_res)
        N_B2 = (1.0 - _get_competitor_possibility(logits_all[5], target_true)).numpy().reshape(grid_res, grid_res)
        N_B3 = (1.0 - _get_competitor_possibility(logits_all[6], target_true)).numpy().reshape(grid_res, grid_res)
        N_B4 = (1.0 - _get_competitor_possibility(logits_all[7], target_true)).numpy().reshape(grid_res, grid_res)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the 4 conclusion canopies
    ax.plot_surface(xx, yy, N_B1, color='red', alpha=0.8, label='Critical Brake')
    ax.plot_surface(xx, yy, N_B2, color='orange', alpha=0.8, label='Prepare to Brake')
    ax.plot_surface(xx, yy, N_B3, color='blue', alpha=0.8, label='Maintain Gap')
    ax.plot_surface(xx, yy, N_B4, color='green', alpha=0.8, label='Safe Cruise')
    
    ax.set_title("EpiMax Rule Base in Autonomous Phase Space", fontsize=16)
    ax.set_xlabel("Relative Distance", fontsize=12)
    ax.set_ylabel("Relative Velocity", fontsize=12)
    ax.set_zlabel("Epistemic Necessity", fontsize=12)
    ax.set_zlim(0, 1)
    
    # Optional: adjust view for the best angle of the "mountain range"
    ax.view_init(elev=25, azim=-60)
    
    plt.tight_layout()
    plt.savefig("results/phase_space_canopies.png", dpi=300)
    print("4-Rule canopy successfully rendered.")

if __name__ == "__main__":
    trained_model = train_model()
    render_phase_space(trained_model)