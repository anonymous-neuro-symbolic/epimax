import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def compute_ood_metrics(id_scores, ood_scores):

    # Check for NaNs before passing to sklearn
    if np.isnan(id_scores).any() or np.isnan(ood_scores).any():
        print("Warning: NaN detected in scores. Replacing with zeros.")
        id_scores = np.nan_to_num(id_scores, nan=0.0)
        ood_scores = np.nan_to_num(ood_scores, nan=0.0)

    """
    Computes standard OOD detection metrics.
    id_scores: Necessity scores (N) for In-Distribution samples.
    ood_scores: Necessity scores (N) for Out-of-Distribution samples.
    """
    # Create labels: 1 for ID, 0 for OOD
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])

    # 1. AUROC (Area Under the Receiver Operating Characteristic)
    auroc = roc_auc_score(labels, scores)

    # 2. AUPR (Area Under the Precision-Recall curve)
    # AUPR-In is typically used for OOD detection
    aupr = average_precision_score(labels, scores)

    # 3. FPR at 95% TPR
    # Determines the False Positive Rate when 95% of ID samples are correctly detected
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # Find the index where TPR is closest to 0.95
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx]

    return {
        "AUROC": auroc,
        "AUPR": aupr,
        "FPR95": fpr95
    }

def evaluate_ood(model, id_loader, ood_loader, device):
    """
    Passes data through the model to extract necessity scores and compute metrics.
    """
    model.eval()
    id_necessities = []
    ood_necessities = []

    with torch.no_grad():
        # Get ID scores
        for images, _ in id_loader:
            images = images.to(device)
            _, _, N = model(images)
            # Take the maximum necessity across classes for each sample
            max_n = torch.max(N, dim=1)[0]
            id_necessities.append(max_n.cpu().numpy())
        
        # Get OOD scores
        for images, _ in ood_loader:
            images = images.to(device)
            _, _, N = model(images)
            max_n = torch.max(N, dim=1)[0]
            ood_necessities.append(max_n.cpu().numpy())

    id_necessities = np.concatenate(id_necessities)
    ood_necessities = np.concatenate(ood_necessities)

    return compute_ood_metrics(id_necessities, ood_necessities)