# src/datasets/synthetic_2d.py
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

class SyntheticBlobs2D(Dataset):
    """
    A 2D synthetic dataset for visualizing decision boundaries 
    and epistemic uncertainty manifolds.
    """
    def __init__(self, n_samples=1500, centers=3, cluster_std=1.2, random_state=42):
        # Generate the raw 2D clusters
        X_np, y_np = make_blobs(
            n_samples=n_samples, 
            centers=centers, 
            n_features=2,
            cluster_std=cluster_std, 
            random_state=random_state
        )
        
        # Standardize features to zero mean and unit variance for stable training
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_np)
        
        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        self.y = torch.tensor(y_np, dtype=torch.long)
        
        # Store bounds for the plotter
        self.x_min, self.x_max = self.X[:, 0].min().item(), self.X[:, 0].max().item()
        self.y_min, self.y_max = self.X[:, 1].min().item(), self.X[:, 1].max().item()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_synthetic_dataloaders(batch_size=64, train_split=0.8):
    """Utility to instantly grab train/val loaders for the synthetic dataset."""
    dataset = SyntheticBlobs2D()
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, dataset