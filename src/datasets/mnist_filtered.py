import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset

def get_filtered_mnist(root='./data', train=True, known_digits=[0,1,2,3,4,5]):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    
    # Identify indices of known vs unknown
    indices = [i for i, label in enumerate(full_dataset.targets) if label in known_digits]
    ood_indices = [i for i, label in enumerate(full_dataset.targets) if label not in known_digits]
    
    return Subset(full_dataset, indices), Subset(full_dataset, ood_indices)