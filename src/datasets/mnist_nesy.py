import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import random

class MNISTAdditionDataset(Dataset):
    def __init__(self, root='./data', train=True, num_digits=2):
        """
        Groups `num_digits` MNIST images together.
        Target is the sum of their true labels.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.mnist = datasets.MNIST(root, train=train, download=True, transform=transform)
        self.num_digits = num_digits
        
        # Pre-calculate lengths to ensure we don't run out of bounds
        self.length = len(self.mnist) // self.num_digits

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        images = []
        total_sum = 0
        
        # Fetch N consecutive images (shuffled dynamically in DataLoader)
        for i in range(self.num_digits):
            actual_idx = (idx * self.num_digits + i) % len(self.mnist)
            img, label = self.mnist[actual_idx]
            images.append(img)
            total_sum += label
            
        # Stack images into shape [num_digits, 1, 28, 28]
        return torch.stack(images), torch.tensor(total_sum, dtype=torch.long)

def get_nesy_loaders(batch_size=128, train_digits=2, eval_digits=[2, 4, 10]):
    train_loader = DataLoader(
        MNISTAdditionDataset(train=True, num_digits=train_digits), 
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    
    eval_loaders = {}
    for n in eval_digits:
        eval_loaders[n] = DataLoader(
            MNISTAdditionDataset(train=False, num_digits=n), 
            batch_size=batch_size, shuffle=False
        )
        
    return train_loader, eval_loaders