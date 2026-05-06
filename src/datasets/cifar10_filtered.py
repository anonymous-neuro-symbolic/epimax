import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset
import numpy as np

class RemappedCIFAR10(Dataset):
    """
    Wraps a CIFAR-10 subset to remap the labels to a continuous 0-N range.
    This is required because cross-entropy and bipolar loss expect contiguous labels.
    """
    def __init__(self, subset, label_mapping):
        self.subset = subset
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, original_label = self.subset[idx]
        # Remap the label. If it's an OOD sample, we can leave it as -1 or original
        # since OOD samples are only used for evaluation, not loss calculation.
        new_label = self.label_mapping.get(original_label, original_label)
        return image, new_label

def get_filtered_cifar10(root='./data', train=True):
    # Standard CIFAR-10 normalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform = transform_train if train else transform_test
    full_dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)

    # CIFAR-10 Classes: 
    # 0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer, 
    # 5: dog, 6: frog, 7: horse, 8: ship, 9: truck

    # We define Vehicles as In-Distribution (ID) and Animals as Out-of-Distribution (OOD)
    id_classes = [0, 1, 8, 9]
    ood_classes = [2, 3, 4, 5, 6, 7]

    # Map ID classes to 0, 1, 2, 3 for the loss function
    label_mapping = {0: 0, 1: 1, 8: 2, 9: 3}

    id_indices = [i for i, label in enumerate(full_dataset.targets) if label in id_classes]
    ood_indices = [i for i, label in enumerate(full_dataset.targets) if label in ood_classes]

    id_subset = Subset(full_dataset, id_indices)
    ood_subset = Subset(full_dataset, ood_indices)

    return RemappedCIFAR10(id_subset, label_mapping), RemappedCIFAR10(ood_subset, label_mapping)