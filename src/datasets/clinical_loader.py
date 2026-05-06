import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class ClinicalDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

def get_clinical_ood_split(csv_path=None, batch_size=64):
    """
    Implements a Population Shift:
    ID: Patients aged 18-60.
    OOD: Patients aged 80+.
    """
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        # Generate synthetic clinical data (Vitals + Demographics)
        np.random.seed(42)
        n = 5000
        data = {
            'heart_rate': np.random.normal(80, 15, n),
            'sys_bp': np.random.normal(120, 20, n),
            'age': np.random.uniform(18, 95, n),
            'label': np.random.randint(0, 2, n) # Mortality
        }
        df = pd.DataFrame(data)

    # Define Covariate Shift
    df_id = df[df['age'] <= 60].copy()
    df_ood = df[df['age'] >= 80].copy()
    
    # Feature scaling (Vital signs must be normalized)
    cols = ['heart_rate', 'sys_bp', 'age']
    for col in cols:
        mu, std = df_id[col].mean(), df_id[col].std()
        df_id[col] = (df_id[col] - mu) / std
        df_ood[col] = (df_ood[col] - mu) / std

    # DataLoaders
    train_ds = ClinicalDataset(df_id[cols].values, df_id['label'].values)
    ood_ds = ClinicalDataset(df_ood[cols].values, df_ood['label'].values)
    
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), \
           DataLoader(ood_ds, batch_size=batch_size, shuffle=False)