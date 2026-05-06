import torch
import torch.optim as optim
from torchvision import datasets, transforms
from src.models.perception_net import PerceptionNet
from src.utils.generic_epicstemoc_logs_deeprpblog_vs_epimax_v2 import generate_epistemic_plots # <-- This is the original import, but we will replace it with the updated version that includes LTN and MAP baselines


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
    # 1. Initialize and Pre-train
    model = PerceptionNet().to(DEVICE)
    pretrain_perception(model)
    
    # 2. Setup testing data for L=15
    print("\nExecuting sensitivity analysis for MNIST-Sum (Length L=15)...")
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=15, shuffle=True, drop_last=True)
    
    # 3. Generate the sensitivity plots (DeepProbLog, MAP, LTN, and 3 EpiMax variants)
    generate_epistemic_plots(model, test_loader, chain_length=15)
    print("Sensitivity analysis complete. Figures saved in results/ directory.")

if __name__ == "__main__":
    main()