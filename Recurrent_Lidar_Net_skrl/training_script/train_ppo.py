"""Training script for PPO algorithm."""
import os
import numpy as np
import torch
import gymnasium as gym

# Import SKRL components for PPO
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

# Import our modules
from gym_interface import make_vector_env
from models import get_model
from utils import seed_everything, create_result_dir, ListDict, ConfigYAML

# import Weights & Biases if enabled in config (no effect if not used)
import wandb

# Wrapper classes for policy and value networks (shared backbone) for PPO
class PolicyWrapper(GaussianMixin, Model):
    """Wrap a base actor-critic network to provide a stochastic policy (Gaussian)."""
    def __init__(self, base, obs_space, act_space, device):
        Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
        GaussianMixin.__init__(self)
        self.base = base
        self.device = next(base.parameters()).device  # ensure same device

    def compute(self, inputs, role):
        # SKRL calls .compute() to get policy distribution parameters
        mean, _ = self.base(inputs['states'])
        # Use a state-independent log_std (expand to match mean shape)
        log_std = self.base.log_std.expand_as(mean)
        return mean, log_std, {}

    def forward(self, x):
        # For direct usage, return mean action (policy mean)
        return self.base(x)[0]

    def set_mode(self, mode: str):
        # Toggle model between train/eval mode
        self.eval() if mode == 'eval' else self.train()

class ValueWrapper(DeterministicMixin, Model):
    """Wrap a base actor-critic network to provide a value function estimator."""
    def __init__(self, base, obs_space, act_space, device):
        Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
        DeterministicMixin.__init__(self)
        self.base = base
        self.device = next(base.parameters()).device

    def compute(self, inputs, role):
        # SKRL calls .compute() for value estimation
        _, value = self.base(inputs['states'])
        return value, {}

    def forward(self, x):
        # For direct usage, return the value estimate
        return self.base(x)[1]

    def set_mode(self, mode: str):
        self.eval() if mode == 'eval' else self.train()

def train(config):
    """Train an agent using Proximal Policy Optimization (PPO)."""
    # Set random seed for reproducibility
    seed_everything(getattr(config, 'seed', 0))

    # Create results directory and save a copy of the config
    results_dir = create_result_dir(config.experiment_name)
    config.save_file(os.path.join(results_dir, 'config.yaml'))

    # Initialize environment(s)
    env = make_vector_env(config)
    device = env.device  # e.g., "cpu" or "cuda"

    # Determine observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = int(np.prod(env.action_space.shape))
    else:
        action_dim = env.action_space.n

    # Initialize base model (shared between policy and value) based on config
    model_type = str(getattr(config, 'model_type', 'basic_lstm')).lower()
    base_model = get_model(model_type, obs_dim, action_dim, config)
    base_model.to(device)

    # Wrap the base model with separate policy and value wrappers (shared parameters)
    models = {
        'policy': PolicyWrapper(base_model, env.observation_space, env.action_space, device),
        'value':  ValueWrapper(base_model, env.observation_space, env.action_space, device)
    }
    models['policy'].to(device)
    models['value'].to(device)

    # Configure PPO agent settings
    cfg_agent = PPO_DEFAULT_CONFIG.copy()
    cfg_agent['discount_factor'] = config.get('gamma', 0.99)
    cfg_agent['lambda'] = config.get('gae_lambda', 0.95)
    cfg_agent['learning_rate'] = config.get('learning_rate', 3e-4)
    cfg_agent['random_timesteps'] = 0  # no random exploration steps at start
    cfg_agent['learning_epochs'] = config.get('ppo_epochs', 4)
    cfg_agent['batch_size'] = config.get('batch_size', 64)

    # Determine number of PPO minibatches from rollout_steps and batch_size
    cfg_agent['mini_batches'] = max(1, int((config.get('rollout_steps', 1024) * env.num_envs) / config.get('batch_size', 64)))
    cfg_agent['rollouts'] = config.get('rollout_steps', 1024)
    cfg_agent['ratio_clip'] = config.get('ppo_clip', 0.2)
    cfg_agent['entropy_loss_scale'] = config.get('entropy_coef', 0.01)

    # Experiment logging and checkpointing configuration
    cfg_agent['experiment']['directory'] = results_dir
    cfg_agent['experiment']['experiment_name'] = config.experiment_name
    cfg_agent['experiment']['wandb'] = config.get('wandb', {}).get('enabled', False)
    if config.get('wandb', {}).get('enabled', False):
        cfg_agent['experiment']['writer'] = 'wandb'
        cfg_agent['experiment']['write_interval'] = config.get('wandb', {}).get('interval', 1000)
        cfg_agent['experiment']['wandb_kwargs'] = {
            'project': config.get('wandb', {}).get('project', None),
            'name': config.get('wandb', {}).get('run_name', None),
            'tags': config.get('wandb', {}).get('tags', [])
        }
    cfg_agent['experiment']['checkpoint_interval'] = config.get('save_interval', 'auto')

    # Set up memory for experience storage (rollout buffer for PPO)
    rollout_steps = cfg_agent['rollouts']
    memory_size = rollout_steps * env.num_envs
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

    # Initialize PPO agent
    agent = PPO(models=models, memory=memory,
                observation_space=env.observation_space, action_space=env.action_space,
                device=device, cfg=cfg_agent)

    # Configure and run the training loop
    cfg_trainer = {
        'timesteps': config.total_timesteps,
        'headless': not getattr(config, 'render', False)  # render flag controls visualization
    }
    trainer = SequentialTrainer(env=env, agents=agent, cfg=cfg_trainer)
    trainer.train()

def main():
    """Parse config and start PPO training."""
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'configs', 'default.yaml')
    cfg = ConfigYAML()
    cfg.load_file(config_path)
    cfg.algorithm = 'ppo'  # ensure correct algorithm field
    train(cfg)

if __name__ == '__main__':
    main()
