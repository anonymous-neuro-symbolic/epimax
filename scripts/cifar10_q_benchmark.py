import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from src.datasets.cifar10_filtered import get_filtered_cifar10
from src.models.centroid_network import create_centroid_resnet18
from src.losses.joint_q_losses import JointQSupremumLoss, q_exp
from src.engine.ood_evaluator import OODEvaluator

def run_q_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = "results/cifar10_q_benchmark"
    os.makedirs(results_dir, exist_ok=True)

    print(">>> Loading Filtered CIFAR-10 Dataset...")
    # 1. Fetch the raw PyTorch Datasets
    # Train = Vehicles (ID), Test OOD = Animals
    train_id_dataset, _ = get_filtered_cifar10(train=True)
    test_id_dataset, test_ood_dataset = get_filtered_cifar10(train=False)
    
    # 2. Wrap them in DataLoaders for batching
    train_id_loader = DataLoader(train_id_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_id_loader = DataLoader(test_id_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_ood_loader = DataLoader(test_ood_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # CIFAR-10 In-Distribution has 4 vehicle classes
    num_classes = 4
    
    print(">>> Initializing Centroid ResNet-18 and Joint q-Supremum Loss...")
    model = create_centroid_resnet18(num_classes=num_classes).to(device)
    
    # We use q=0.5 for compact support, and lambda_n=1.0 for the necessity constraint
    q_value = 0.5
    criterion = JointQSupremumLoss(q=q_value, lambda_n=1.0)
    
    # Standard Adam optimizer and Cosine Annealing
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    # --- Training Phase ---
    print("\n--- Starting Training ---")
    num_epochs = 40
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_id_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch}/{num_epochs-1}] - Avg Loss: {running_loss/len(train_id_loader):.4f}")

    # Save model weights
    torch.save(model.state_dict(), f"{results_dir}/q_supremum_resnet18.pth")

    # --- Evaluation Phase ---
    print("\n--- Evaluating Epistemic Uncertainty ---")
    model.eval()
    
    def get_uncertainty_scores(loader):
        uncertainties = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                logits = model(images)
                
                # Transform negative distances to possibility: pi = exp_q(z)
                pi = q_exp(logits, q=q_value)
                
                # Epistemic Uncertainty = 1.0 - max(pi)
                # Represents total ignorance if no class centroid matches the features
                batch_uncert = 1.0 - torch.max(pi, dim=1)[0]
                uncertainties.append(batch_uncert.cpu().numpy())
                
        return np.concatenate(uncertainties)

    print("Extracting In-Distribution (Vehicles) uncertainty...")
    id_scores = get_uncertainty_scores(test_id_loader)
    
    print("Extracting Out-of-Distribution (Animals) uncertainty...")
    ood_scores = get_uncertainty_scores(test_ood_loader)

   # --- Metric Calculation (Self-Contained) ---
    def calculate_metrics(id_uncert, ood_uncert):
        y_true = np.concatenate([np.zeros(len(id_uncert)), np.ones(len(ood_uncert))])
        y_scores = np.concatenate([id_uncert, ood_uncert])
        
        auroc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        idx = np.where(tpr >= 0.95)[0][0]
        fpr95 = fpr[idx]
        
        return {"auroc": auroc, "aupr": aupr, "fpr95": fpr95}

    results = calculate_metrics(id_scores, ood_scores)
    
    print("\n=== Final Benchmark Results ===")
    print(f"AUROC: {results['auroc']:.4f}")
    print(f"AUPR:  {results['aupr']:.4f}")
    print(f"FPR95: {results['fpr95']:.4f}")

    # Save metrics
    with open(f"{results_dir}/metrics.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    # Generate Density Plot natively
    plt.figure(figsize=(8, 6))
    plt.hist(id_scores, bins=50, alpha=0.6, density=True, label='ID (Vehicles)', color='green')
    plt.hist(ood_scores, bins=50, alpha=0.6, density=True, label='OOD (Animals)', color='red')
    plt.title('Epistemic Uncertainty Density: Joint $q$-Supremum')
    plt.xlabel(r'Epistemic Uncertainty $U = 1 - \max \pi$')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.savefig(f"{results_dir}/q_supremum_density.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"Artifacts saved to {results_dir}/")

if __name__ == "__main__":
    run_q_benchmark()