import torch
import yaml
import argparse
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.models.bipolar_network import create_bipolar_model
from src.losses.bipolar_loss import BipolarPossibilisticLoss
from src.engine.trainer import BipolarTrainer

def get_cifar100_superclass_map(dataset):
    """
    Extracts the mapping from 100 fine labels to 20 coarse labels.
    Required for HVR calculation.
    """
    mapping = torch.zeros(100, dtype=torch.long)
    # CIFAR100 object contains 'targets' (fine) and 'coarse_targets' (superclass)
    for fine, coarse in zip(dataset.targets, dataset.coarse_targets):
        mapping[fine] = coarse
    return mapping

def run_experiment(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Data loading (Optimized for V100 32GB)
    # If on V100, we can use higher resolution or larger batches
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Better feature extraction for Wide-ResNet
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    train_set = datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
    test_set = datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    superclass_map = get_cifar100_superclass_map(train_set).to(device)

    # 2. Model & Loss initialization
    model = create_bipolar_model(
        model_name=config['model'],
        num_pos=100,
        pretrained=config.get('pretrained', True)
    )

    loss_fn = BipolarPossibilisticLoss(
        gamma=config['gamma'],
        lambda_1=config['l1'],
        lambda_2=config['l2']
    )

    # 3. Training setup
    trainer = BipolarTrainer(model, loss_fn, train_loader, test_loader, device, config)
    
    best_acc = 0.0
    best_hvr = 1.0 # Lower is better
    final_delta = 0.0
    
    # 4. Training loop
    for epoch in range(config['epochs']):
        train_loss = trainer.train_epoch(epoch)
        acc, hvr = trainer.evaluate(is_hierarchical=True, superclass_map=superclass_map)
        
        # Track the best metrics for the sweep runner
        if acc > best_acc:
            best_acc = acc
            best_hvr = hvr # We care about the HVR at the point of highest accuracy
            
        print(f" * Validation -> Acc: {acc:.4f} | HVR: {hvr:.4f}")

    # Final metrics for the CSV
    return {
        "best_accuracy": best_acc,
        "best_hvr": best_hvr,
        "final_loss": train_loss
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    results = run_experiment(args.config)
    print(f"\nExperiment finished. Best Accuracy: {results['best_accuracy']:.4f}")