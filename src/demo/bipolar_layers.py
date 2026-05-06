# epimax/bipolar_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceptionNet(nn.Module):
    """
    Standard LeNet-5 style architecture for MNIST digit perception.
    Outputs raw pre-activation logits required by the EpiMax loss formulation.
    """
    def __init__(self, num_classes=10):
        super(PerceptionNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        logits = self.fc2(x)
        return logits