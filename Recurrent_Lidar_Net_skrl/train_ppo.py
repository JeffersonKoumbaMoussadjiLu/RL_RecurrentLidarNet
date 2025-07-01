import os, sys, yaml
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import wandb
#from skrl.agents.torch import DQN, DQN_DEFAULT_CONFIG, PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG          # DQN / DDQN
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG          # PPO / PPO_RNN

from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

import gymnasium as gym
from gym_interface import make_vector_env
from models import get_model
from utils import seed_everything, create_result_dir


# Wrap the base model to create separate policy and value models (shared parameters)
# Use GaussianMixin for stochastic policy (continuous actions) => predict the expected return for taking a specific action in a given state
# Use DeterministicMixin for value function (state value estimation) => predict the expected return from a state, regardless of the action taken
class PolicyWrapper(GaussianMixin, Model):
    #def __init__(self, base):
    
    def __init__(self, base, obs_space, act_space, device):  

        # initialise BOTH parents ─ Model first, then GaussianMixin
        # intialise Model with observation and action spaces
        Model.__init__(self,
                    observation_space=obs_space,
                    action_space=act_space,
                    device=device,
        ) 

        GaussianMixin.__init__(self)
                
        self.base = base                        # shared backbone       
        self.device = next(base.parameters()).device   # ensure same device

        #super().__init__()
        #self.base = base
        #self.device = next(base.parameters()).device   # ensure same device

    # skrl will call .compute()
    def compute(self, inputs, role):
        #skrl calls this instead of forward()
        # inputs["states"] is a tensor already on self.device
        mean, _ = self.base(inputs["states"])
        log_std        = self.base.log_std.expand_as(mean)   # same σ for every state
        return mean, log_std, {}      # GaussianMixin needs (μ, logσ)
        

    def forward(self, x):
        return self.base(x)[0] # return only action_mean
        #action_mean, _ = self.base(x)
        #return action_mean

    # Set training/evaluation mode
    def set_mode(self, mode: str):
        self.eval() if mode == "eval" else self.train()
        
class ValueWrapper(DeterministicMixin, Model):
    #def __init__(self, base):
    #    super().__init__()
    #    self.base = base
    #    self.device = next(base.parameters()).device   # ensure same device

    def __init__(self, base, obs_space, act_space, device):
        Model.__init__(self,
            observation_space = obs_space,
            action_space = act_space,
            device = device,
        )

        DeterministicMixin.__init__(self)  # initialise DeterministicMixin
                    
        self.base = base
        self.device = next(base.parameters()).device   # ensure same device


    def forward(self, x):
        return self.base(x)[1] # return only state_value
        #_, state_value = self.base(x)
        #return state_value

    def compute(self, inputs, role):
        _, value = self.base(inputs["states"])
        return value, {}         # tensor first, then a dict

    def set_mode(self, mode: str):
        self.eval() if mode == "eval" else self.train()

def main():
    # path to yaml file
    config_path = Path(__file__).resolve().parent / "configs" / "default.yaml"

    # Load configuration from YAML
    config_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 \
                else config_path

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed for reproducibility
    seed_everything(config.get("seed", 0))

    # Create results directory and save a copy of the config
    results_dir = create_result_dir(config["experiment_name"])
    with open(os.path.join(results_dir, "config.yaml"), 'w') as f:
        yaml.safe_dump(config, f)

    # Initialize environment(s)
    env = make_vector_env(config)
    device = env.device  # device used by environment ("cpu" or "cuda")

    # Configure agent based on selected algorithm
    algorithm = config["algorithm"].lower()

    # On-policy PPO configuration
    obs_dim = env.observation_space.shape[0]

    #action_dim = env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else env.action_space.n

    # works for every Box (and for Discrete as well)
    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = int(np.prod(env.action_space.shape))
    else:
        action_dim = env.action_space.n

    # Select neural network architecture for policy/value
    model_type = config.get("model_type", "basic_lstm").lower()

    
    base_model = get_model(model_type, obs_dim, action_dim, config).to(device)

    models = {
    "policy": PolicyWrapper(
        base_model,
        env.observation_space,
        env.action_space,
        device
    ),

    "value":  ValueWrapper(
        base_model,
        env.observation_space,
        env.action_space,
        device
    )
    }

    #models["policy"] = PolicyWrapper(base_model)
    #models["value"] = ValueWrapper(base_model)
    models["policy"].to(device)
    models["value"].to(device)

    # Configure PPO agent settings
    cfg_agent = PPO_DEFAULT_CONFIG.copy()
    cfg_agent["discount_factor"] = config.get("gamma", 0.99)
    cfg_agent["lambda"] = config.get("gae_lambda", 0.95)
    cfg_agent["learning_rate"] = config.get("learning_rate", 3e-4)
    cfg_agent["random_timesteps"] = 0
    cfg_agent["learning_epochs"] = config.get("ppo_epochs", 4)

    # Calculate number of mini-batches from rollout size and batch_size
    cfg_agent["mini_batches"] = max(1, int((config.get("rollout_steps", 1024) * env.num_envs) / config.get("batch_size", 64)))
    cfg_agent["rollouts"] = config.get("rollout_steps", 1024)
    cfg_agent["ratio_clip"] = config.get("ppo_clip", 0.2)
    cfg_agent["entropy_loss_scale"]     = config.get("entropy_coef", 0.01)
    
    # Experiment logging and checkpointing
    cfg_agent["experiment"]["directory"] = results_dir
    cfg_agent["experiment"]["experiment_name"] = config["experiment_name"]
    cfg_agent["experiment"]["wandb"] = config.get("wandb", {}).get("enabled", False)
    if config.get("wandb", {}).get("enabled", False):
        cfg_agent["experiment"]["writer"] = "wandb"                  
        cfg_agent["experiment"]["write_interval"] = config["wandb"].get("interval", 1000)  
        cfg_agent["experiment"]["wandb_kwargs"] = {
            "project": config["wandb"]["project"],
            "name": config["wandb"]["run_name"],
            "tags"   : config["wandb"].get("tags", [])
        }
    cfg_agent["experiment"]["checkpoint_interval"] = config.get("save_interval", "auto")

    # Memory size must be a multiple of (rollout_steps * num_envs)
    rollout_len  = cfg_agent["rollouts"]                  # e.g. 1024
    memory_size  = rollout_len * env.num_envs             # one slot per (env, step)# Set up rollout storage (memory)

    # Set up rollout storage (memory)
    memory = RandomMemory(                               
        memory_size = memory_size,
        num_envs     = env.num_envs,
        device       = device
    )   

    # Initialize PPO agent
    agent = PPO(models=models,
                memory=memory,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device,
                cfg=cfg_agent)
    
    # Set up and run the training process
    cfg_trainer = {
        "timesteps": config["total_timesteps"],
        "headless": True  # no rendering
    }
    trainer = SequentialTrainer(env=env, agents=agent, cfg=cfg_trainer)
    trainer.train()

if __name__ == "__main__":
    main()


# ["speed", "steering_angle"]
# ["accl", "steering_speed"]