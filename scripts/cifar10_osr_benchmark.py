import torch
import torch.optim as optim
import json
import os
from src.datasets.cifar10_filtered import get_filtered_cifar10
from src.models.baselines import SoftmaxBaseline
from src.models.bipolar_network import create_bipolar_model
from src.losses.bipolar_loss import BipolarPossibilisticLoss
from src.engine.ood_evaluator import OODEvaluator
from torch.utils.data import DataLoader

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = "results/cifar10_osr"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load Filtered Data (ID: 4 Vehicles, OOD: 6 Animals)
    train_id, _ = get_filtered_cifar10(train=True)
    test_id, test_ood = get_filtered_cifar10(train=False)
    
    train_loader = DataLoader(train_id, batch_size=128, shuffle=True)
    id_test_loader = DataLoader(test_id, batch_size=128, shuffle=False)
    ood_test_loader = DataLoader(test_ood, batch_size=128, shuffle=False)

    methods = ['Softmax', 'Bipolar']
    final_metrics = {}

    for method in methods:
        print(f"\n>>> Training {method} CIFAR-10 Benchmark...")
        
        if method == 'Softmax':
            # 4 known vehicle classes
            model = SoftmaxBaseline(num_classes=4)
            # Adapt the baseline backbone for CIFAR 32x32
            model.backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.backbone.maxpool = torch.nn.Identity()
            model = model.to(device)
            criterion = torch.nn.CrossEntropyLoss()
            is_bipolar = False
        else:
            # 4 known vehicle classes
            model = create_bipolar_model('resnet18_cifar', num_pos=4).to(device)
            criterion = BipolarPossibilisticLoss(gamma=2.0, lambda_1=1.0, lambda_2=1.0)
            is_bipolar = True

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # 2. Training Loop (40 Epochs for CIFAR-10)
        for epoch in range(40):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                
                outputs = model(images)
                if is_bipolar:
                    loss_tuple = criterion(outputs[0], outputs[1], labels)
                    loss = loss_tuple[0] if isinstance(loss_tuple, tuple) else loss_tuple
                else:
                    loss = criterion(outputs, labels)
                    
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            print(f"Epoch {epoch} complete. Avg Loss: {running_loss/len(train_loader):.4f}")

        # 3. Evaluation
        print(f"Evaluating {method}...")
        evaluator = OODEvaluator(model, device)
        metrics = evaluator.evaluate_ood(id_test_loader, ood_test_loader, is_bipolar=is_bipolar)
        
        evaluator.plot_density(metrics, f"{results_dir}/{method}_density.pdf")
        
        final_metrics[method] = {
            "AUROC": float(metrics['auroc']),
            "AUPR": float(metrics['aupr']),
            "FPR95": float(metrics['fpr95'])
        }

    # 4. Save report
    with open(f"{results_dir}/cifar10_benchmark_results.json", 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    print("\nBenchmark Finished! Results in results/cifar10_osr/")

if __name__ == "__main__":
    run_benchmark()