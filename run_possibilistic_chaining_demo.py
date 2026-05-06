from src.demo.bipolar_layers import PerceptionNet
from src.demo.possibilistic_chain import evaluate_chain_necessity
import torch
import numpy as np


def run_stability_demonstration(chain_lengths=[2, 5, 10, 15, 20], num_samples=1000):
    print("EpiMax Epistemic Stability Demonstration")
    print("Target: Demonstrate non-vanishing confidence over long reasoning chains")
    print("-" * 55)
    print(f"{'Logical Depth (L)':<20} | {'Oracle Acc':<12} | {'EpiMax Necessity':<15}")
    print("-" * 55)
    
    model = PerceptionNet()
    model.eval()

    with torch.no_grad():
        for L in chain_lengths:
            poss_confs = []
            oracle_correct = 0
            
            for _ in range(num_samples):
                logits_chain = []
                targets_chain = []
                chain_is_correct = True
                
                for _ in range(L):
                    # Simulate standard neural perception outputs
                    simulated_logits = torch.randn(1, 10) * 2.0 
                    target = torch.randint(0, 10, (1,))
                    
                    if np.random.rand() > 0.5:
                        simulated_logits[0, target] += 4.0 
                    else:
                        chain_is_correct = False
                        
                    logits_chain.append(simulated_logits)
                    targets_chain.append(target)
                
                if chain_is_correct:
                    oracle_correct += 1
                
                # Evaluate solely the possibilistic bound
                pos_conf = evaluate_chain_necessity(logits_chain, targets_chain)
                poss_confs.append(pos_conf.item())
                
            acc = (oracle_correct / num_samples) * 100
            avg_poss = np.median(poss_confs)
            
            print(f"{L:<20} | {acc:>9.1f}% | {avg_poss:>15.4f}")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    run_stability_demonstration()