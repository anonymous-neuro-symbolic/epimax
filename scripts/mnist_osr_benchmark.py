import torch
import torch.optim as optim
import json
import os
from src.datasets.mnist_filtered import get_filtered_mnist
from src.models.baselines import SoftmaxBaseline
from src.models.bipolar_network import create_bipolar_model # Your model
from src.losses.bipolar_loss import BipolarPossibilisticLoss
from src.engine.ood_evaluator import OODEvaluator
from torch.utils.data import DataLoader

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = "results/mnist_osr"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load Filtered Data (0-5 Known, 6-9 OOD)
    train_id, _ = get_filtered_mnist(train=True)
    test_id, test_ood = get_filtered_mnist(train=False)
    
    train_loader = DataLoader(train_id, batch_size=128, shuffle=True)
    id_test_loader = DataLoader(test_id, batch_size=128, shuffle=False)
    ood_test_loader = DataLoader(test_ood, batch_size=128, shuffle=False)

    methods = ['Softmax', 'Bipolar']
    final_metrics = {}

    for method in methods:
        print(f"\n>>> Training {method} Benchmark...")
        
        if method == 'Softmax':
            model = SoftmaxBaseline(num_classes=6).to(device)
            criterion = torch.nn.CrossEntropyLoss()
            is_bipolar = False
        else:
            model = create_bipolar_model('resnet18_mnist', num_pos=6).to(device)
            criterion = BipolarPossibilisticLoss(gamma=2.0, lambda_1=1.0, lambda_2=1.0)
            is_bipolar = True

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # 2. Training Loop (15 Epochs is enough for MNIST)
        for epoch in range(15):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                
                outputs = model(images)
                if is_bipolar:
                    # Bipolar model returns (pi_plus, pi_minus)
                    # The criterion returns a tuple (total_loss, loss_plus, loss_minus)
                    # We only need the first element for backpropagation
                    loss_tuple = criterion(outputs[0], outputs[1], labels)
                    loss = loss_tuple[0] if isinstance(loss_tuple, tuple) else loss_tuple                    
                else:
                    loss = criterion(outputs, labels)
                    
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch} complete.")

        # 3. Evaluation using our OODEvaluator
        evaluator = OODEvaluator(model, device)
        metrics = evaluator.evaluate_ood(id_test_loader, ood_test_loader, is_bipolar=is_bipolar)
        
        # Save plots for the paper
        evaluator.plot_density(metrics, f"{results_dir}/{method}_density.pdf")
        
        # Store numbers
        final_metrics[method] = {
            "AUROC": float(metrics['auroc']),
            "AUPR": float(metrics['aupr']),
            "FPR95": float(metrics['fpr95'])
        }

    # 4. Save final JSON report
    with open(f"{results_dir}/benchmark_results.json", 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    print("\nBenchmark Finished! Results in results/mnist_osr/")

if __name__ == "__main__":
    run_benchmark()