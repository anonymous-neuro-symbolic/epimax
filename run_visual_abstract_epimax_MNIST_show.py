import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. Perception Network
class PerceptionNet(nn.Module):
    def __init__(self):
        super(PerceptionNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# 2. EpiMax Loss
class EpiMaxLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=2.0):
        super(EpiMaxLoss, self).__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def forward(self, logits, targets):
        batch_size = logits.shape[0]
        z_target = logits[torch.arange(batch_size), targets]
        z_max_all, _ = torch.max(logits, dim=1)
        
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(batch_size), targets] = False
        z_max_comp, _ = torch.max(logits.masked_fill(~mask, float('-inf')), dim=1)
        
        pi_target = torch.exp(z_target - z_max_all)
        loss_poss = self.alpha * torch.pow(torch.clamp(1.0 - pi_target, min=1e-7), self.gamma)
        
        N_bar = torch.exp(z_max_comp - z_max_all)
        loss_nec = self.beta * torch.pow(N_bar, self.gamma)
        return torch.mean(loss_poss + loss_nec)

# 3. Compute EpiMax for ALL 10 Classes
def compute_epimax_all_classes(logits):
    """Calculates Pi and N for every single class (0-9)"""
    z_max, _ = torch.max(logits, dim=1, keepdim=True)
    pi = torch.exp(logits - z_max) # Feasibility of all classes
    
    N = torch.zeros_like(logits)
    for i in range(logits.shape[1]):
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, i] = False
        z_comp, _ = torch.max(logits.masked_fill(~mask, float('-inf')), dim=1)
        # N(y) = 1 - Pi(competitor)
        N[:, i] = torch.clamp(1.0 - torch.exp(z_comp - z_max.squeeze()), min=0.0)
    return pi, N

# 4. Main Execution
def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PerceptionNet().to(device)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fashion_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=128, shuffle=True)
    
    print("Training anchored network (3 Epochs, SGD + Weight Decay)...")
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
    
    criterion = EpiMaxLoss()
    
    model.train()
    for epoch in range(3):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
    
    model.eval()
    
    # Extract Scenarios
    print("Preparing the 3 Scenarios...")
    img_8, img_1, img_3, img_shoe = None, None, None, None
    for img, label in mnist_test:
        if label == 8 and img_8 is None: img_8 = img
        if label == 1 and img_1 is None: img_1 = img
        if label == 3 and img_3 is None: img_3 = img
        if img_8 is not None and img_3 is not None: break
        
    for img, label in fashion_test:
        if label == 9: # Ankle boot
            img_shoe = img
            break
            
    # Create the ambiguous overlap (Conflict)
    img_ambiguous = 0.4 * img_8 + 0.7 * img_3
    

    
    scenarios = [
        ("Clear Digit '8'", img_8),
        ("Ambiguous '3' vs '8'", img_ambiguous),
        ("OOD Shoe", img_shoe)
    ]
    
    # 5. Plotting the 2D Cartesian Spaces
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (title, img) in enumerate(scenarios):
        ax = axes[idx]
        
        # Inference
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(device))
            pi, N = compute_epimax_all_classes(logits)
            pi = pi.squeeze().cpu().numpy()
            N = N.squeeze().cpu().numpy()
            
        # Draw background zones
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5) # N = Pi Boundary
        ax.fill_between([0, 1], [0, 1], 1.1, color='gray', alpha=0.2) # Axiomatic Void
        ax.fill_between([0.5, 1.1], 0.5, 1.1, color='green', alpha=0.1) # Certainty
        ax.fill_between([0.5, 1.1], -0.1, 0.5, color='orange', alpha=0.1) # Ignorance/Conflict
        ax.fill_between([-0.1, 0.5], -0.1, 0.5, color='blue', alpha=0.05) # Rejection
        
        # Annotate zones
        ax.text(0.2, 0.8, 'Axiomatic Void\n($N > \Pi$)', color='black', alpha=0.6, ha='center', fontsize=16, weight='bold')
        ax.text(0.8, 0.8, 'Certainty', color='green', alpha=0.6, ha='center', fontsize=18, weight='bold')
        ax.text(0.8, 0.2, 'Ignorance /\nConflict', color='orange', alpha=0.8, ha='center', fontsize=18, weight='bold')
        ax.text(0.2, 0.2, 'Rejection', color='blue', alpha=0.5, ha='center', fontsize=18, weight='bold')

        # Plot the 10 classes
        scatter = ax.scatter(pi, N, c=range(10), cmap='tab10', s=100, edgecolors='k', zorder=5)
        
        # Add text labels for the classes so you can see where '3' and '8' land
        for i in range(10):
            # Add a slight jitter for overlapping points in the rejection zone
            jitter_y = np.random.uniform(-0.02, 0.02) if N[i] < 0.05 else 0
            ax.annotate(str(i), (pi[i], N[i] + jitter_y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=16, weight='bold')

        ax.set_title(title, fontsize=22, weight='bold')
        ax.set_xlabel(r'Possibility ($\Pi$)', fontsize=20)
        if idx == 0:
            ax.set_ylabel(r'Necessity ($N$)', fontsize=20)
            
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    plt.suptitle("EpiMax Epistemic Space: 10-Class Distribution Analysis", fontsize=26, weight='bold', y=1.05)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()