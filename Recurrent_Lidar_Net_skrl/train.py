"""Main training script that dispatches to specific algorithm trainers based on config."""
import sys
from pathlib import Path
from utils import ConfigYAML

def main():
    """Main entry point for training. Load config and dispatch to algorithm-specific training."""
    # Determine configuration file path (default to 'configs/default.yaml' in the current directory)
    here = Path(__file__).resolve().parent
    default_config_path = here / 'configs' / 'default.yaml'
    config_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else default_config_path

    # Load configuration
    config = ConfigYAML()
    config.load_file(config_path)
    algorithm = getattr(config, 'algorithm', None)
    if algorithm is None:
        raise ValueError("Config file missing 'algorithm' field.")
    algorithm = str(algorithm).lower()

    # Dispatch to the appropriate training module
    if algorithm == 'ppo':
        from training_script import train_ppo
        train_ppo.train(config)

    elif algorithm == 'dqn':
        from training_script import train_dqn
        train_dqn.train(config)
        
    else:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Please implement train_{algorithm}.py or add handling in train.py.")

if __name__ == "__main__":
    main()
