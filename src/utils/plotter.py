# src/utils/plotter.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch.nn.functional as F

def plot_epistemic_boundaries(model, epimax_layer, dataset, device='cpu', resolution=0.05, save_path=None):
    """
    Generates a 1x3 NeurIPS-style figure comparing the decision boundary, 
    the Possibility manifold, and the Necessity margin.
    """
    model.eval()
    epimax_layer.eval()

    # 1. Create a dense 2D meshgrid spanning the dataset bounds + padding
    padding = 2.0
    xx, yy = np.meshgrid(
        np.arange(dataset.x_min - padding, dataset.x_max + padding, resolution),
        np.arange(dataset.y_min - padding, dataset.y_max + padding, resolution)
    )

    # 2. Forward pass the entire grid through the model
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(grid_tensor)
        pi, N = epimax_layer(logits)
        
        # Get the winning class and its corresponding pi/N values
        pi_max, predictions = torch.max(pi, dim=1)
        n_max, _ = torch.max(N, dim=1)

    # 3. Reshape the 1D results back into 2D grid format for plotting
    Z_pred = predictions.cpu().numpy().reshape(xx.shape)
    Z_pi = pi_max.cpu().numpy().reshape(xx.shape)
    Z_N = n_max.cpu().numpy().reshape(xx.shape)

    # Extract raw data points for overlay
    X_raw = dataset.X.numpy()
    y_raw = dataset.y.numpy()

    # 4. Set up the plotting aesthetics (NeurIPS style)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap_custom = ListedColormap(['#FF9999', '#99FF99', '#99CCFF']) # Soft colors for classes

    # Subplot 1: Standard Decision Boundary
    axes[0].contourf(xx, yy, Z_pred, alpha=0.4, cmap=cmap_custom)
    axes[0].scatter(X_raw[:, 0], X_raw[:, 1], c=y_raw, s=20, edgecolor='k', cmap=cmap_custom)
    axes[0].set_title(r'Standard Decision Boundary ($\arg\max z$)', fontsize=14)
    
    # Subplot 2: Possibility Manifold
    # Using a continuous colormap (viridis) to show the decay from 1.0 (certain) to 0.0
    contour_pi = axes[1].contourf(xx, yy, Z_pi, levels=20, cmap='viridis', alpha=0.8)
    axes[1].scatter(X_raw[:, 0], X_raw[:, 1], c='white', s=10, edgecolor='k', alpha=0.5)
    axes[1].set_title(r'Possibility Surface ($\max \pi$)', fontsize=14)
    fig.colorbar(contour_pi, ax=axes[1], fraction=0.046, pad=0.04)

    # Subplot 3: Necessity Margin
    # Shows the strict safe zones bounded by the dual loss
    contour_N = axes[2].contourf(xx, yy, Z_N, levels=20, cmap='plasma', alpha=0.8)
    axes[2].scatter(X_raw[:, 0], X_raw[:, 1], c='white', s=10, edgecolor='k', alpha=0.5)
    axes[2].set_title(r'Necessity Margin ($\max N$)', fontsize=14)
    fig.colorbar(contour_N, ax=axes[2], fraction=0.046, pad=0.04)

    # Clean up axes
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved strictly to {save_path}")
    
    plt.show()

def plot_comparative_boundaries(model_poss, epimax_layer, model_edl, dataset, device='cpu', save_path=None):
    """Plots a 2x2 comparison: Possibilistic Framework vs Evidential Deep Learning."""
    model_poss.eval()
    epimax_layer.eval()
    model_edl.eval()

    resolution = 0.05
    padding = 2.0
    xx, yy = np.meshgrid(
        np.arange(dataset.x_min - padding, dataset.x_max + padding, resolution),
        np.arange(dataset.y_min - padding, dataset.y_max + padding, resolution)
    )
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

    # --- Evaluate Possibilistic Model ---
    with torch.no_grad():
        logits_poss = model_poss(grid_tensor)
        pi, N = epimax_layer(logits_poss)
        Z_pred_poss = torch.argmax(logits_poss, dim=1).cpu().numpy().reshape(xx.shape)
        Z_N = torch.max(N, dim=1)[0].cpu().numpy().reshape(xx.shape)

    # --- Evaluate Evidential (EDL) Model ---
    with torch.no_grad():
        logits_edl = model_edl(grid_tensor)
        evidence = F.relu(logits_edl)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        u = (3.0 / S).squeeze() # K=3 classes
        
        Z_pred_edl = torch.argmax(logits_edl, dim=1).cpu().numpy().reshape(xx.shape)
        Z_unc_edl = u.cpu().numpy().reshape(xx.shape)

    # --- Plotting ---
    X_raw, y_raw = dataset.X.numpy(), dataset.y.numpy()
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    cmap_custom = ListedColormap(['#FF9999', '#99FF99', '#99CCFF'])

    # 1. Possibilistic Decision Boundary
    axes[0, 0].contourf(xx, yy, Z_pred_poss, alpha=0.4, cmap=cmap_custom)
    axes[0, 0].scatter(X_raw[:, 0], X_raw[:, 1], c=y_raw, s=20, edgecolor='k', cmap=cmap_custom)
    axes[0, 0].set_title(r'Ours: Decision Boundary ($\arg\max z$)', fontsize=14)

    # 2. Possibilistic Necessity Margin
    contour_N = axes[0, 1].contourf(xx, yy, Z_N, levels=20, cmap='plasma', alpha=0.8)
    axes[0, 1].scatter(X_raw[:, 0], X_raw[:, 1], c='white', s=10, edgecolor='k', alpha=0.5)
    axes[0, 1].set_title(r'Ours: Necessity Margin ($\max N$)', fontsize=14)
    fig.colorbar(contour_N, ax=axes[0, 1])

    # 3. EDL Decision Boundary
    axes[1, 0].contourf(xx, yy, Z_pred_edl, alpha=0.4, cmap=cmap_custom)
    axes[1, 0].scatter(X_raw[:, 0], X_raw[:, 1], c=y_raw, s=20, edgecolor='k', cmap=cmap_custom)
    axes[1, 0].set_title(r'EDL: Decision Boundary', fontsize=14)

    # 4. EDL Epistemic Uncertainty
    contour_edl = axes[1, 1].contourf(xx, yy, Z_unc_edl, levels=20, cmap='magma_r', alpha=0.8)
    axes[1, 1].scatter(X_raw[:, 0], X_raw[:, 1], c='white', s=10, edgecolor='k', alpha=0.5)
    axes[1, 1].set_title(r'EDL: Epistemic Uncertainty ($u = K/S$)', fontsize=14)
    fig.colorbar(contour_edl, ax=axes[1, 1])

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_triple_comparison(model_poss, epimax_layer, model_edl, model_duq, dataset, device='cpu', save_path=None):
    model_poss.eval(); epimax_layer.eval(); model_edl.eval(); model_duq.eval()

    res, pad = 0.05, 2.0
    xx, yy = np.meshgrid(
        np.arange(dataset.x_min - pad, dataset.x_max + pad, res),
        np.arange(dataset.y_min - pad, dataset.y_max + pad, res)
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

    with torch.no_grad():
        # 1. Possibilistic
        l_poss = model_poss(grid)
        _, N = epimax_layer(l_poss)
        Z_p_poss = torch.argmax(l_poss, 1).cpu().numpy().reshape(xx.shape)
        Z_N = torch.max(N, 1)[0].cpu().numpy().reshape(xx.shape)
        
        # 2. EDL
        l_edl = model_edl(grid)
        u_edl = (3.0 / (torch.sum(F.relu(l_edl) + 1.0, 1)))
        Z_p_edl = torch.argmax(l_edl, 1).cpu().numpy().reshape(xx.shape)
        Z_u_edl = u_edl.cpu().numpy().reshape(xx.shape)
        
        # 3. DUQ
        rbf_q = model_duq(grid)
        Z_p_duq = torch.argmax(rbf_q, 1).cpu().numpy().reshape(xx.shape)
        Z_u_duq = (1.0 - torch.max(rbf_q, 1)[0]).cpu().numpy().reshape(xx.shape)

    X_raw, y_raw = dataset.X.numpy(), dataset.y.numpy()
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    cmap_c = ListedColormap(['#FF9999', '#99FF99', '#99CCFF'])

    def draw_row(row, Z_p, Z_unc, title_p, title_u, cmap_u):
            # Left column: Decision boundary with class-colored dots
            axes[row, 0].contourf(xx, yy, Z_p, alpha=0.3, cmap=cmap_c)
            axes[row, 0].scatter(X_raw[:, 0], X_raw[:, 1], c=y_raw, s=15, edgecolor='k', cmap=cmap_c)
            axes[row, 0].set_title(title_p, fontsize=13)
            
            # Right column: Epistemic metric (Necessity/Uncertainty)
            c_u = axes[row, 1].contourf(xx, yy, Z_unc, levels=20, cmap=cmap_u, alpha=0.8)
            
            # ADD THIS LINE: Overlay data points on the uncertainty/necessity heatmaps
            # We use white dots for better visibility against colorful heatmaps
            axes[row, 1].scatter(X_raw[:, 0], X_raw[:, 1], c='white', s=10, edgecolor='k', alpha=0.6)
            
            axes[row, 1].set_title(title_u, fontsize=13)
            fig.colorbar(c_u, ax=axes[row, 1])

    draw_row(0, Z_p_poss, Z_N, r'Ours: Decision boundary', r'Ours: Necessity margin ($\max N$)', 'plasma')
    draw_row(1, Z_p_edl, Z_u_edl, r'EDL: Decision boundary', r'EDL: Epistemic uncertainty ($u$)', 'magma_r')
    draw_row(2, Z_p_duq, Z_u_duq, r'DUQ: Decision boundary', r'DUQ: Distance-based uncertainty', 'inferno')

    for ax in axes.flatten():
        ax.set_xticks([]); ax.set_yticks([])
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_epimax_internals(model, epimax_layer, dataset, device='cpu', save_path=None):
    """
    Plots the internal possibilistic state: 
    Necessity (Certainty) vs. Competitor Possibility (Ambiguity).
    """
    model.eval(); epimax_layer.eval()
    res, pad = 0.05, 2.0
    xx, yy = np.meshgrid(
        np.arange(dataset.x_min - pad, dataset.x_max + pad, res),
        np.arange(dataset.y_min - pad, dataset.y_max + pad, res)
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(grid)
        pi, N = epimax_layer(logits)
        
        # 1. Necessity of the winner (Certainty)
        Z_N = torch.max(N, dim=1)[0].cpu().numpy().reshape(xx.shape)
        
        # 2. Possibility of the strongest competitor (Ambiguity)
        # We sort pi to find the second highest value
        top2_pi, _ = torch.topk(pi, k=2, dim=1)
        Z_pi_sec = top2_pi[:, 1].cpu().numpy().reshape(xx.shape)
        
        # 3. Combined Metric: The Epistemic Gap (Conflict)
        # Higher values mean the model sees multiple plausible but contradictory answers
        Z_conflict = (Z_pi_sec * (1 - Z_N)).reshape(xx.shape)

    X_raw, y_raw = dataset.X.numpy(), dataset.y.numpy()
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Necessity Plot (Certainty)
    c1 = axes[0].contourf(xx, yy, Z_N, levels=20, cmap='plasma', alpha=0.8)
    axes[0].scatter(X_raw[:, 0], X_raw[:, 1], c='white', s=10, edgecolor='k', alpha=0.6)
    axes[0].set_title(r'Necessity ($N$): Certainty of Leader', fontsize=18)
    fig.colorbar(c1, ax=axes[0])

    # Competitor Possibility (Ambiguity)
    c2 = axes[1].contourf(xx, yy, Z_pi_sec, levels=20, cmap='viridis', alpha=0.8)
    axes[1].scatter(X_raw[:, 0], X_raw[:, 1], c='white', s=10, edgecolor='k', alpha=0.6)
    axes[1].set_title(r'Competitor $\pi$: Degree of Ambiguity', fontsize=18)
    fig.colorbar(c2, ax=axes[1])

    # Combined: Conflict
    c3 = axes[2].contourf(xx, yy, Z_conflict, levels=20, cmap='hot', alpha=0.8)
    axes[2].scatter(X_raw[:, 0], X_raw[:, 1], c='white', s=10, edgecolor='k', alpha=0.6)
    axes[2].set_title(r'Combined: Possibilistic Conflict', fontsize=18)
    fig.colorbar(c3, ax=axes[2])

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.show()


def plot_variance_heatmaps(stats, dataset, xx, yy, save_path=None):
    """
    Visualizes the standard deviation (instability) of uncertainty manifolds.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Mapping keys to ensure they match the main script
    v_epimax = stats['epimax_n'].reshape(xx.shape)
    v_edl = stats['edl'].reshape(xx.shape)
    v_duq = stats['duq'].reshape(xx.shape)
    
    vmax = max(v_epimax.max(), v_edl.max(), v_duq.max())

    def draw_var_plot(ax, data, title, cmap):
        c = ax.contourf(xx, yy, data, levels=30, cmap=cmap, vmin=0, vmax=vmax)
        ax.scatter(dataset.X[:,0], dataset.X[:,1], c='white', s=5, alpha=0.3)
        ax.set_title(title, fontsize=14)
        return c

    draw_var_plot(axes[0], v_epimax, "EpiMax: Necessity variance", "inferno")
    draw_var_plot(axes[1], v_edl, "EDL: Uncertainty variance", "inferno")
    c_final = draw_var_plot(axes[2], v_duq, "DUQ: Uncertainty variance", "inferno")
    
    fig.colorbar(c_final, ax=axes.ravel().tolist(), label="Standard deviation ($\sigma$)")
    
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()

def plot_gamma_sensitivity(save_path=None):
    """
    Replicates the focal loss behavior plot for the Dual Possibilistic Loss.
    """
    s = np.linspace(0.001, 1.0, 500)
    gammas = [0, 0.5, 1, 2, 5]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Possibility term (Feasibility)
    for g, c in zip(gammas, colors):
        loss_pi = (1 - s)**g
        ax1.plot(s, loss_pi, label=rf'$\gamma = {g}$', color=c, lw=2.5)
    
    ax1.set_title("Possibility term focal modulation", fontsize=18)
    ax1.set_xlabel(r"Target possibility ($\pi_{y^*}$)", fontsize=18)
    ax1.set_ylabel("Loss contribution", fontsize=18)
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    ax1.legend()
    
    # 2. Necessity term (Certainty)
    for g, c in zip(gammas, colors):
        loss_n = (s)**g
        ax2.plot(s, loss_n, label=rf'$\gamma = {g}$', color=c, lw=2.5)
    
    ax2.set_title("Necessity term focal modulation", fontsize=18)
    ax2.set_xlabel(r"Competitor possibility ($\pi_{max \neq y^*}$)", fontsize=18)
    ax2.set_ylabel("Loss contribution", fontsize=18)
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
    plt.show()