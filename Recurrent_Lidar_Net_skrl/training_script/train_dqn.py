"""Training script for DQN algorithm (Deep Q-Network)."""
import os
import numpy as np
import torch
import gymnasium as gym

# Import SKRL components for DQN
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.models.torch import Model, DeterministicMixin

# Import project modules
from gym_interface import make_vector_env
from models import DuelingTransformerNet
from utils import seed_everything, create_result_dir, ListDict, ConfigYAML

# Optional Weights & Biases import
import wandb

class DuelingQNet(DeterministicMixin, Model):
    """Wrap DuelingTransformerNet for use as a Q-network in SKRL (dueling DQN architecture)."""
    def __init__(self, obs_space, act_space, device):
        Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
        DeterministicMixin.__init__(self)
        # Determine input and output dimensions
        obs_dim = int(np.prod(obs_space.shape))
        act_dim = act_space.n if hasattr(act_space, 'n') else int(np.prod(act_space.shape))
        # Initialize backbone Q-network (dueling architecture)
        self.backbone = DuelingTransformerNet(obs_dim, act_dim).to(device)
        self.device = device

    def compute(self, inputs, role):
        # Called by SKRL to compute Q-values from state
        q_values = self.backbone(inputs['states'])
        return q_values, {}

    def forward(self, x):
        # Standard forward to get Q-values
        return self.backbone(x)

    def set_mode(self, mode: str):
        self.eval() if mode == 'eval' else self.train()

def train(config):
    """Train an agent using Deep Q-Network (DQN) algorithm."""
    # Set seed for reproducibility
    seed_everything(getattr(config, 'seed', 0))

    # Create results directory and save config
    results_dir = create_result_dir(config.experiment_name)
    config.save_file(os.path.join(results_dir, 'config.yaml'))

    # Initialize environment(s)
    env = make_vector_env(config)
    device = env.device

    # Determine observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n if hasattr(env.action_space, 'n') else int(np.prod(env.action_space.shape))

    # Initialize Q-network and target Q-network using dueling architecture
    models = {
        'q_network': DuelingQNet(env.observation_space, env.action_space, device),
        'target_q_network': DuelingQNet(env.observation_space, env.action_space, device)
    }
    models['q_network'].to(device)
    models['target_q_network'].to(device)

    # Set up experience replay memory
    memory = RandomMemory(memory_size=config.get('replay_buffer_size', 100000), num_envs=env.num_envs, device=device)

    # Configure DQN agent settings
    cfg_agent = DQN_DEFAULT_CONFIG.copy()
    cfg_agent['discount_factor'] = config.get('gamma', 0.99)
    cfg_agent['batch_size'] = config.get('batch_size', 64)
    cfg_agent['exploration']['initial_epsilon'] = config.get('epsilon_start', 1.0)
    cfg_agent['exploration']['final_epsilon'] = config.get('epsilon_end', 0.05)
    cfg_agent['exploration']['timesteps'] = config.get('epsilon_decay', 10000)
    
    # Logging and checkpoint configuration
    cfg_agent['experiment']['directory'] = results_dir
    cfg_agent['experiment']['experiment_name'] = config.experiment_name
    cfg_agent['experiment']['wandb'] = config.get('wandb', {}).get('enabled', False)
    if config.get('wandb', {}).get('enabled', False):
        cfg_agent['experiment']['writer'] = 'wandb'
        # Use save_interval for logging frequency (or 'auto')
        cfg_agent['experiment']['write_interval'] = config.get('save_interval', 'auto')
        cfg_agent['experiment']['wandb_kwargs'] = {
            'project': config.get('wandb', {}).get('project', None),
            'name': config.get('wandb', {}).get('run_name', None),
            'tags': config.get('wandb', {}).get('tags', [])
        }
    cfg_agent['experiment']['checkpoint_interval'] = config.get('save_interval', 'auto')

    # Initialize DQN agent
    agent = DQN(models=models, memory=memory,
                observation_space=env.observation_space, action_space=env.action_space,
                device=device, cfg=cfg_agent)

    # Train the agent
    cfg_trainer = {
        'timesteps': config.total_timesteps,
        'headless': not getattr(config, 'render', False)
    }
    trainer = SequentialTrainer(env=env, agents=agent, cfg=cfg_trainer)
    trainer.train()

def main():
    """Parse config and start DQN training."""
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'configs', 'default.yaml')
    cfg = ConfigYAML()
    cfg.load_file(config_path)
    cfg.algorithm = 'dqn'
    train(cfg)

if __name__ == '__main__':
    main()
