import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from src.models.perception_net import PerceptionNet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def pretrain_perception():
    model = PerceptionNet().to(DEVICE)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=256, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    print("Pre-training PerceptionNet for 1 epoch...")
    for img, label in train_loader:
        img, label = img.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(img), label)
        loss.backward()
        optimizer.step()
    return model

def main():
    model = pretrain_perception()
    model.eval()

    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    
    # 1. Generate a batch of pure OOD data (Maximum Gaussian Noise)
    num_samples = 1000
    noise_variance = 1.5
    
    noisy_images = []
    for i in range(num_samples):
        img, _ = test_dataset[i]
        noisy_img = img + noise_variance * torch.randn_like(img)
        noisy_img = torch.clamp(noisy_img, 0., 1.)
        noisy_img = transforms.functional.normalize(noisy_img, (0.1307,), (0.3081,))
        noisy_images.append(noisy_img)
        
    noisy_batch = torch.stack(noisy_images).to(DEVICE)

    # 2. Extract logits once (Massive speedup)
    print("\nExtracting logits for OOD samples...")
    with torch.no_grad():
        logits = model(noisy_batch) # Shape: [1000, 10]
        z_max, _ = torch.max(logits, dim=1, keepdim=True) # Shape: [1000, 1]

    # 3. Define the fine-grained parameter grid
    grid_size = 40
    alphas = np.linspace(0.1, 5.0, grid_size)
    betas = np.linspace(0.1, 5.0, grid_size)
    
    # Matrix to store average Necessity for the heatmap
    heatmap_data = np.zeros((grid_size, grid_size))

    print(f"Evaluating {grid_size * grid_size} parameter configurations...")
    
    # 4. Vectorized parameter sweep
    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            # Calculate possibilities with current alpha
            pi = torch.exp((logits - z_max) / alpha)
            
            # Mask out the winner to find the top competitor
            pred_idx = pi.argmax(dim=1)
            mask = torch.ones_like(pi, dtype=torch.bool)
            mask[torch.arange(num_samples), pred_idx] = False
            
            competitors = pi.masked_fill(~mask, 0.0)
            max_competitor_pi, _ = competitors.max(dim=1)
            
            # Calculate Necessity with current beta
            N = torch.clamp(1.0 - (beta * max_competitor_pi), min=0.0, max=1.0)
            
            # Store the mean Necessity across all 1000 noisy samples
            heatmap_data[i, j] = N.mean().item()

    # --- Plotting the Heatmap ---
    sns.set_theme(style="white")
    plt.figure(figsize=(10, 8))
    
    # We flip the data so higher Betas are at the top of the Y-axis
    ax = sns.heatmap(np.flipud(heatmap_data), cmap="YlOrRd_r", 
                     xticklabels=np.round(alphas, 1), 
                     yticklabels=np.round(np.flip(betas), 1),
                     cbar_kws={'label': 'Average Necessity (Lower is better for OOD)'})

    # Clean up the ticks so it's not overly crowded
    ax.set_xticks(ax.get_xticks()[::4])
    ax.set_xticklabels(np.round(alphas, 1)[::4])
    ax.set_yticks(ax.get_yticks()[::4])
    ax.set_yticklabels(np.round(np.flip(betas), 1)[::4])

    plt.title("EpiMax epistemic boundary mapping (Pure noise state)", fontsize=16)
    plt.xlabel(r"Possibility temperature ($\alpha$)", fontsize=14)
    plt.ylabel(r"Conflict penalty ($\beta$)", fontsize=14)
    plt.tight_layout()
    
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "epimax_parameter_heatmap.png"), dpi=300)
    print(f"\nHeatmap successfully generated in {results_dir}")

if __name__ == "__main__":
    main()