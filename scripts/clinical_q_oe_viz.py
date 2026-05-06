import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.datasets.clinical_loader import get_clinical_ood_split
from src.models.tabular_centroid_network import TabularCentroidNet
from src.losses.joint_q_losses import JointQSupremumLoss, q_exp
from src.losses.outlier_penalties import active_repulsive_loss
from scripts.clinical_q_benchmark2 import extract_bipolar_metrics

def plot_uncertainty_heatmap(model, device, q_val, save_path):
    """
    Generates a 2D heatmap of the uncertainty manifold.
    X-axis: Systolic BP, Y-axis: Heart Rate (normalized units).
    """
    model.eval()
    # Create a grid across normalized physiological space
    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-3, 3, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    
    # Age is fixed at 'Adult' level (e.g., 0.0 in normalized units)
    grid_data = np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())]
    grid_tensor = torch.tensor(grid_data, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(grid_tensor)
        pi = q_exp(logits, q=q_val)
        uncertainty = 1.0 - torch.max(pi, dim=1)[0]
        zz = uncertainty.cpu().numpy().reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(xx, yy, zz, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Epistemic Uncertainty $U$')
    plt.title('Clinical Manifold Dynamics: Uncertainty Heatmap')
    plt.xlabel('Normalized systolic BP')
    plt.ylabel('Normalized heart rate')
    
    # Mark the learnable centroids in the projection
    # (Note: Centroids are in 64D latent space, so we visualize the input field)
    plt.savefig(save_path)
    plt.close()

def run_visual_refinement():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = "results/clinical_oe_viz"
    os.makedirs(results_dir, exist_ok=True)
    
    train_loader, ood_loader = get_clinical_ood_split(batch_size=64)
    model = TabularCentroidNet(input_dim=3, num_classes=2).to(device)
    
    q_val = 0.5
    lambda_oe = 0.1 # Reduced lambda to prevent the 100%/100% collapse
    criterion_id = JointQSupremumLoss(q=q_val)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(">>> Training with active boundary refinement (Phase B)...")
    for epoch in range(51):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            # Background noise for Outlier Exposure
            bg_data = torch.randn_like(features) * 2.0 
            
            optimizer.zero_grad()
            l_id = criterion_id(model(features), labels)
            l_oe = active_repulsive_loss(model(bg_data), q=q_val)
            
            (l_id + lambda_oe * l_oe).backward()
            optimizer.step()
        
        if epoch % 25 == 0:
            plot_uncertainty_heatmap(model, device, q_val, 
                                     f"{results_dir}/heatmap_epoch_{epoch}.pdf")

    print(f"Dynamics visualizations saved to {results_dir}/")

if __name__ == "__main__":
    run_visual_refinement()