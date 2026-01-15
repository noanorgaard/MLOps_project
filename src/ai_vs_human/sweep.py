"""Launch a WandB hyperparameter sweep."""
import wandb
import yaml
from pathlib import Path

def launch_sweep():
    """Initialize and run a hyperparameter sweep."""
    # Load sweep configuration
    sweep_config_path = Path("configs/sweep_config.yaml")
    with open(sweep_config_path) as f:
        sweep_config = yaml.safe_load(f)
    
    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="MLOps_project"
    )
    
    print(f"Sweep initialized with ID: {sweep_id}")
    print(f"Run agents with: wandb agent {sweep_id}")
    
    return sweep_id

if __name__ == "__main__":
    launch_sweep()
