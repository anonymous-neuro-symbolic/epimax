import torch
import torch.optim as optim
import json
import os
from src.datasets.cifar10_filtered import get_filtered_cifar10
from src.models.bipolar_network import create_bipolar_model
from src.losses.bipolar_loss import BipolarPossibilisticLoss
from src.engine.ood_evaluator import OODEvaluator
from torch.utils.data import DataLoader

def run_lambda_sweep():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = "results/cifar10_lambda_sweep"
    os.makedirs(results_dir, exist_ok=True)

    train_id, _ = get_filtered_cifar10(train=True)
    test_id, test_ood = get_filtered_cifar10(train=False)
    
    train_loader = DataLoader(train_id, batch_size=128, shuffle=True)
    id_test_loader = DataLoader(test_id, batch_size=128, shuffle=False)
    ood_test_loader = DataLoader(test_ood, batch_size=128, shuffle=False)

    lambda_1_values = [1.0, 0.1, 0.01, 0.0]
    sweep_metrics = {}

    for l1 in lambda_1_values:
        run_name = f"Bipolar_L1_{l1}"
        print(f"\n>>> Starting Sweep: {run_name}")
        
        # 4 known vehicle classes
        model = create_bipolar_model('resnet18_cifar', num_pos=4).to(device)
        
        # Keep gamma and lambda_2 constant, only vary lambda_1
        criterion = BipolarPossibilisticLoss(gamma=2.0, lambda_1=l1, lambda_2=1.0)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # 30 Epochs is sufficient to observe manifold separation
        for epoch in range(30):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                
                outputs = model(images)
                loss_tuple = criterion(outputs[0], outputs[1], labels)
                loss = loss_tuple[0] if isinstance(loss_tuple, tuple) else loss_tuple
                    
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            print(f"Epoch {epoch} | Avg Loss: {running_loss/len(train_loader):.4f}")

        print(f"Evaluating {run_name}...")
        evaluator = OODEvaluator(model, device)
        metrics = evaluator.evaluate_ood(id_test_loader, ood_test_loader, is_bipolar=True)
        
        evaluator.plot_density(metrics, f"{results_dir}/{run_name}_density.pdf")
        
        sweep_metrics[run_name] = {
            "AUROC": float(metrics['auroc']),
            "AUPR": float(metrics['aupr']),
            "FPR95": float(metrics['fpr95'])
        }

        # Save incrementally
        with open(f"{results_dir}/sweep_results.json", 'w') as f:
            json.dump(sweep_metrics, f, indent=4)

    print("\nSweep Complete! Check results/cifar10_lambda_sweep/")

if __name__ == "__main__":
    run_lambda_sweep()