import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader

class RemappedDataset(Dataset):
    """
    Generalizes label remapping for any subset.
    """
    def __init__(self, subset, label_mapping):
        self.subset = subset
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        return image, self.label_mapping.get(label, label)

def get_cifar10_ood_loaders(root='./data', batch_size=128, mode='dataset'):
    """
    Supports two modes:
    'semantic': Vehicles (ID) vs Animals (OOD) within CIFAR-10.
    'dataset': Full CIFAR-10 (ID) vs SVHN (OOD).
    """
    # Standard CIFAR-10 Normalization
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm
    ])
    
    transform_test = transforms.Compose([transforms.ToTensor(), norm])

    if mode == 'semantic':
        full_train = datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
        full_test = datasets.CIFAR10(root, train=False, download=True, transform=transform_test)
        
        # Define Vehicles as ID (4 classes)
        id_classes = [0, 1, 8, 9] 
        label_mapping = {0: 0, 1: 1, 8: 2, 9: 3}
        
        # Filter indices
        id_train_idx = [i for i, l in enumerate(full_train.targets) if l in id_classes]
        id_test_idx = [i for i, l in enumerate(full_test.targets) if l in id_classes]
        ood_test_idx = [i for i, l in enumerate(full_test.targets) if l not in id_classes]

        train_loader = DataLoader(RemappedDataset(Subset(full_train, id_train_idx), label_mapping), batch_size=batch_size, shuffle=True)
        id_test_loader = DataLoader(RemappedDataset(Subset(full_test, id_test_idx), label_mapping), batch_size=batch_size)
        ood_test_loader = DataLoader(Subset(full_test, ood_test_idx), batch_size=batch_size)
        
    else: # 'dataset' mode: CIFAR-10 vs SVHN
        train_set = datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10(root, train=False, download=True, transform=transform_test)
        
        # SVHN as OOD (Requires resizing to 32x32 if not already)
        ood_set = datasets.SVHN(root, split='test', download=True, transform=transform_test)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        id_test_loader = DataLoader(test_set, batch_size=batch_size)
        ood_test_loader = DataLoader(ood_set, batch_size=batch_size)

    return train_loader, id_test_loader, ood_test_loader