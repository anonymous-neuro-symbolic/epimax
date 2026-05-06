import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader

class RemappedDataset(Dataset):
    """
    Wraps a dataset subset to remap labels to a continuous 0-N range.
    """
    def __init__(self, subset, label_mapping):
        self.subset = subset
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, original_label = self.subset[idx]
        new_label = self.label_mapping.get(original_label, original_label)
        return image, new_label

def get_cifar10_ood_loaders(root='./data', batch_size=512, ood_type='svhn'):
    """
    Optimized for V100. 
    ID is always the full CIFAR-10.
    OOD is either 'svhn' (Far-OOD) or 'cifar100' (Near-OOD).
    """
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm
    ])
    
    transform_test = transforms.Compose([transforms.ToTensor(), norm])

    # In-Distribution: CIFAR-10
    train_set = datasets.CIFAR10(root, train=True, download=False, transform=transform_train)
    test_set = datasets.CIFAR10(root, train=False, download=False, transform=transform_test)

    # Out-of-Distribution choice
    if ood_type == 'svhn':
        ood_set = datasets.SVHN(root, split='test', download=False, transform=transform_test)
    elif ood_type == 'cifar100':
        ood_set = datasets.CIFAR100(root, train=False, download=False, transform=transform_test)
    else:
        raise ValueError("Invalid ood_type. Choose 'svhn' or 'cifar100'.")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    id_loader = DataLoader(test_set, batch_size=batch_size, pin_memory=True, num_workers=4)
    ood_loader = DataLoader(ood_set, batch_size=batch_size, pin_memory=True, num_workers=4)

    return train_loader, id_loader, ood_loader