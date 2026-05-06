import os
import yaml
import pandas as pd
from main import run_experiment # Ensure run_experiment returns the best metrics

def conduct_param_sweep():
    # 1. Define the grid based on our theoretical priorities
    gammas = [1.0, 2.0, 4.0]
    lambda_2_ratios = [0.5, 1.0, 2.0] # Ratio of l2/l1 (Necessity strength)
    
    base_config_path = 'configs/exp1_hierarchical.yaml'
    results_dir = 'results/sweep_data'
    os.makedirs(results_dir, exist_ok=True)
    
    sweep_results = []

    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    for g in gammas:
        for l_ratio in lambda_2_ratios:
            # Derive specific lambdas
            l1 = 1.0
            l2 = l1 * l_ratio
            
            run_name = f"gamma_{g}_lratio_{l_ratio}"
            print(f"\n{'='*40}\nSTARTING RUN: {run_name}\n{'='*40}")
            
            # Update config for this specific run
            current_config = base_config.copy()
            current_config['gamma'] = float(g)
            current_config['l1'] = float(l1)
            current_config['l2'] = float(l2)
            current_config['experiment_name'] = run_name
            
            # Save temporary config for the main script
            temp_config_path = 'configs/temp_sweep_config.yaml'
            with open(temp_config_path, 'w') as f:
                yaml.dump(current_config, f)
            
            try:
                # Execute training and retrieve peak performance metrics
                # Note: You should modify main.py's run_experiment to return these
                metrics = run_experiment(temp_config_path)
                
                metrics.update({
                    'gamma': g,
                    'l2_l1_ratio': l_ratio,
                    'status': 'SUCCESS'
                })
                sweep_results.append(metrics)
                
            except Exception as e:
                print(f"Run {run_name} failed: {e}")
                sweep_results.append({'gamma': g, 'l2_l1_ratio': l_ratio, 'status': f'FAILED: {str(e)}'})

            # Incremental save in case the V100 session is interrupted
            pd.DataFrame(sweep_results).to_csv('results/sweep_summary.csv', index=False)

    print("\nSweep complete. Data saved to results/sweep_summary.csv")

if __name__ == "__main__":
    conduct_param_sweep()