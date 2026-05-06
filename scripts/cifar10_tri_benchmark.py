import torch
import torch.optim as optim
import torch.nn.functional as F
import json
import os
from src.datasets.cifar10_filtered import get_filtered_cifar10
from src.models.baselines import SoftmaxBaseline
from src.models.bipolar_network import create_bipolar_model
from src.losses.bipolar_loss import BipolarPossibilisticLoss
from src.losses.edl_loss import EDLLoss
from src.engine.ood_evaluator import OODEvaluator
from torch.utils.data import DataLoader
import numpy as np

def run_tri_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = "results/cifar10_tri_benchmark"
    os.makedirs(results_dir, exist_ok=True)

    train_id, _ = get_filtered_cifar10(train=True)
    test_id, test_ood = get_filtered_cifar10(train=False)
    
    train_loader = DataLoader(train_id, batch_size=128, shuffle=True)
    id_test_loader = DataLoader(test_id, batch_size=128, shuffle=False)
    ood_test_loader = DataLoader(test_ood, batch_size=128, shuffle=False)

    methods = ['Softmax', 'EDL', 'Bipolar']
    final_metrics = {}

    for method in methods:
        print(f"\n>>> Training {method} CIFAR-10 Benchmark...")
        num_classes = 4
        
        if method == 'Softmax':
            model = SoftmaxBaseline(num_classes=num_classes)
            model.backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.backbone.maxpool = torch.nn.Identity()
            model = model.to(device)
            criterion = torch.nn.CrossEntropyLoss()
            
        elif method == 'EDL':
            # EDL uses the exact same backbone architecture as Softmax
            model = SoftmaxBaseline(num_classes=num_classes)
            model.backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.backbone.maxpool = torch.nn.Identity()
            model = model.to(device)
            criterion = EDLLoss(num_classes=num_classes, annealing_step=10)
            
        else: # Bipolar
            model = create_bipolar_model('resnet18_cifar', num_pos=num_classes).to(device)
            # Using the optimal lambda values from our previous sweep diagnostics
            criterion = BipolarPossibilisticLoss(gamma=2.0, lambda_1=0.01, lambda_2=1.0)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Training Loop
        for epoch in range(40):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                
                outputs = model(images)
                
                if method == 'Bipolar':
                    loss_tuple = criterion(outputs[0], outputs[1], labels)
                    loss = loss_tuple[0] if isinstance(loss_tuple, tuple) else loss_tuple
                elif method == 'EDL':
                    # EDL requires strictly positive evidence
                    evidence = F.softplus(outputs)
                    loss = criterion(evidence, labels, epoch)
                else:
                    loss = criterion(outputs, labels)
                    
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            if epoch % 5 == 0 or epoch == 39:
                print(f"Epoch {epoch} complete. Avg Loss: {running_loss/len(train_loader):.4f}")

        # Custom Evaluation Logic
        print(f"Evaluating {method}...")
        model.eval()
        
        def get_scores(loader):
            all_scores = []
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(device)
                    outputs = model(images)
                    
                    if method == 'Bipolar':
                        pi_plus = outputs[0]
                        top2_pi = torch.topk(pi_plus, 2, dim=1)[0]
                        scores = top2_pi[:, 1]
                    elif method == 'EDL':
                        evidence = F.softplus(outputs)
                        alpha = evidence + 1.0
                        S = torch.sum(alpha, dim=1)
                        # Vacuity: u = K / S
                        scores = num_classes / S
                    else: # Softmax
                        probs = torch.softmax(outputs, dim=1)
                        scores = 1.0 - torch.max(probs, dim=1)[0]
                        
                    all_scores.append(scores.cpu().numpy())
            return np.concatenate(all_scores)

        id_scores = get_scores(id_test_loader)
        ood_scores = get_scores(ood_test_loader)
        
        # We temporarily bypass the OODEvaluator class to directly use our custom get_scores
        # You will need to extract the AUROC/AUPR/FPR95 math from your evaluator here, 
        # or update your OODEvaluator to handle the EDL vacuity metric.
        evaluator = OODEvaluator(model, device)
        results = evaluator.calculate_metrics(id_scores, ood_scores)
        evaluator.plot_density({'id_scores': id_scores, 'ood_scores': ood_scores}, f"{results_dir}/{method}_density.pdf")
        
        final_metrics[method] = {
            "AUROC": float(results['auroc']),
            "AUPR": float(results['aupr']),
            "FPR95": float(results['fpr95'])
        }

    with open(f"{results_dir}/tri_benchmark_results.json", 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    print("\nPhase 1 Complete! Check results/cifar10_tri_benchmark/")

if __name__ == "__main__":
    run_tri_benchmark()