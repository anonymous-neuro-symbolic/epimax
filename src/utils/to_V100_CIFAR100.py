import torchvision
# Download CIFAR-100 to the local ./data folder
torchvision.datasets.CIFAR100(root='./data', train=False, download=True)