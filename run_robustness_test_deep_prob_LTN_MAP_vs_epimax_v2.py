import torch
import torch.optim as optim
from torchvision import datasets, transforms
from src.models.perception_net import PerceptionNet

# Import the updated suite of visualization and evaluation utilities
from src.utils.generic_epicstemoc_logs_deeprpblog_vs_epimax_v3 import (
    visualize_sample_logits,
    generate_accuracy_across_lengths    
)
# Import the updated suite of visualization and evaluation utilities
from src.utils.generic_epicstemoc_logs_deeprpblog_vs_epimax_v2 import (
    generate_epistemic_plots
)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def pretrain_perception(model):
    """Establish a high-accuracy baseline for perception."""
    print("Pre-training Perception Layer on MNIST...")
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=128, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for img, label in train_loader:
        img, label = img.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(img), label)
        loss.backward()
        optimizer.step()

def main():
    # 1. Initialize and Pre-train Perception
    model = PerceptionNet().to(DEVICE)
    pretrain_perception(model)
    
    # 2. Setup testing data
    print("\nLoading MNIST test dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)
    
    # Standard loader for static sequence tasks (L=15) and logit extraction
    standard_test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=15, shuffle=True, drop_last=True
    )
    
    # ---------------------------------------------------------
    # Visual Abstract Generation Pipeline
    # ---------------------------------------------------------
    
    # Step A: Visualize raw energy landscapes (Pre-activation Logits)
    print("\nGenerating sample logit visualizations (Energy Landscape context)...")
    visualize_sample_logits(model, standard_test_loader, device=DEVICE)
    
    # Step B: Evaluate multi-length deductive accuracy (Panels A & C)
    # Passes the raw dataset so the function can dynamically adjust sequence lengths L
    print("\nExecuting multi-length extrapolation accuracy analysis (L=2 to 20)...")
    generate_accuracy_across_lengths(model, test_dataset, tau=100.0, device=DEVICE)
    
    # Step C: Generate Epistemic Breakdown and Calibration (Original L=15 test)
    print("\nExecuting epistemic breakdown and calibration for chain length L=15...")
    generate_epistemic_plots(model, standard_test_loader, chain_length=15, tau=100.0)

    print("\nAll empirical analyses complete. Check the 'results/' directory for publication-ready panels.")

if __name__ == "__main__":
    main()