import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

class OODEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def get_scores(self, loader, is_bipolar=True):
        """
        Extracts the 'Uncertainty Score' for OOD detection.
        For Bipolar: Score = 1 - max(pi_plus)
        For Softmax: Score = 1 - max(softmax_probs)
        """
        self.model.eval()
        all_scores = []
        
        for images, _ in loader:
            images = images.to(self.device)            
            if is_bipolar:
                pi_plus, _ = self.model(images)
                # In Possibility Theory, max(pi) is ALWAYS 1.0. 
                # Uncertainty is the "Lack of Necessity", measured by the strength 
                # of the strongest alternative hypothesis.
                top2_pi = torch.topk(pi_plus, 2, dim=1)[0]
                # Score is the possibility of the second most likely class
                scores = top2_pi[:, 1] 
            else:
                logits = self.model(images)
                probs = torch.softmax(logits, dim=1)
                # In Probability, uncertainty is the lack of confidence in the top class
                scores = 1.0 - torch.max(probs, dim=1)[0]
            
            # if is_bipolar:
            #     pi_plus, _ = self.model(images)
            #     # Uncertainty is the lack of possibility in the known manifold
            #     scores = 1.0 - torch.max(pi_plus, dim=1)[0]
            # else:
            #     logits = self.model(images)
            #     probs = torch.softmax(logits, dim=1)
            #     scores = 1.0 - torch.max(probs, dim=1)[0]
            
            all_scores.append(scores.cpu().numpy())
            
        return np.concatenate(all_scores)

    def evaluate_ood(self, id_loader, ood_loader, is_bipolar=True):
        """
        Computes AUROC, AUPR, and FPR@95.
        """
        id_scores = self.get_scores(id_loader, is_bipolar)
        ood_scores = self.get_scores(ood_loader, is_bipolar)
        
        # Labels: 0 for In-Distribution, 1 for Out-of-Distribution
        y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        y_scores = np.concatenate([id_scores, ood_scores])
        
        auroc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)
        
        # Calculate FPR at 95% TPR
        fpr95 = self.calculate_fpr95(id_scores, ood_scores)
        
        return {
            "auroc": auroc,
            "aupr": aupr,
            "fpr95": fpr95,
            "id_scores": id_scores,
            "ood_scores": ood_scores
        }

    def calculate_fpr95(self, id_scores, ood_scores):
        # Find the threshold where 95% of ID samples are correctly classified as ID
        threshold = np.percentile(id_scores, 95)
        # Calculate how many OOD samples fall below this threshold (False Positives)
        fpr = np.mean(ood_scores < threshold)
        return fpr

    def plot_density(self, results, save_path):
        plt.figure(figsize=(10, 6))
        sns.kdeplot(results['id_scores'], label='In-Distribution (0-5)', fill=True, bw_adjust=0.5)
        sns.kdeplot(results['ood_scores'], label='Out-of-Distribution (6-9)', fill=True, bw_adjust=0.5)
        plt.title("Possibility density distribution for OOD detection")
        plt.xlabel("Uncertainty Score ($1 - \max \pi^+$)")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(save_path)
        plt.close()