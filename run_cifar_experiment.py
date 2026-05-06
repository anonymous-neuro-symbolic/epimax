import torch
import pandas as pd
import os
import time
from src.datasets.cifar_datasets import get_cifar10_ood_loaders
from src.models.epimax_resnet import EpiMaxResNet18
from src.losses.epimax_loss import DualPossibilisticLoss
from src.engine.trainer import train_epimax_resnet
from src.engine.evaluator import evaluate_ood

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GAMMAS = [0, 0.5, 1.0, 2.0, 5.0]
K_ROUNDS = 10
EPOCHS = 100
BATCH_SIZE = 512

def main():
    os.makedirs('results', exist_ok=True)
    all_results = []

    for gamma in GAMMAS:
        for r in range(K_ROUNDS):
            print(f"\n[INIT] Gamma: {gamma} | Round: {r+1}/{K_ROUNDS}")
            
            # 1. Always train on CIFAR-10
            # We use SVHN loader initially just to get the train_loader
            train_loader, id_loader, svhn_loader = get_cifar10_ood_loaders(batch_size=BATCH_SIZE, ood_type='svhn')
            _, _, c100_loader = get_cifar10_ood_loaders(batch_size=BATCH_SIZE, ood_type='cifar100')

            model = EpiMaxResNet18(num_classes=10).to(DEVICE)
            criterion = DualPossibilisticLoss(gamma=gamma)

            # 2. Training
            model = train_epimax_resnet(model, train_loader, criterion, DEVICE, epochs=EPOCHS)

            # 3. Dual Evaluation
            print("Evaluating Far-OOD (SVHN)...")
            svhn_metrics = evaluate_ood(model, id_loader, svhn_loader, DEVICE)
            
            print("Evaluating Near-OOD (CIFAR-100)...")
            c100_metrics = evaluate_ood(model, id_loader, c100_loader, DEVICE)

            # 4. Data Logging
            for ood_name, m in [("SVHN", svhn_metrics), ("CIFAR100", c100_metrics)]:
                all_results.append({
                    "gamma": gamma,
                    "round": r + 1,
                    "ood_dataset": ood_name,
                    **m
                })
            
            pd.DataFrame(all_results).to_csv('results/dual_ood_sweep_partial.csv', index=False)

    pd.DataFrame(all_results).to_csv('results/dual_ood_sweep_final.csv', index=False)
    print("\nDual experiment complete. Data saved to results/dual_ood_sweep_final.csv")

if __name__ == "__main__":
    main()