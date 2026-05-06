import torchvision

# Download CIFAR10 to the local ./data folder
torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# Download SVHN to the local ./data folder
torchvision.datasets.SVHN(root='./data', split='test', download=True)