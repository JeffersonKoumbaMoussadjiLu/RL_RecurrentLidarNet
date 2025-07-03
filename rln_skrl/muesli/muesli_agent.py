import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class MuesliAgent:
    def __init__(self, env, representation, dynamics, prediction, config, device):
        self.env = env
        self.representation = representation
        self.dynamics = dynamics
        self.prediction = prediction
        self.config = config
        self.device = device

        self.optimizer = torch.optim.Adam(
            list(self.representation.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.prediction.parameters()),
            lr=config.get("learning_rate", 3e-4)
        )

        self.memory = deque(maxlen=config.get("buffer_size", 10000))
        self.batch_size = config.get("batch_size", 32)
        self.support_size = 10
        self.gamma = config.get("gamma", 0.99)

    def select_action(self, state, epsilon=0.1):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            latent = self.representation(state)
            logits, _ = self.prediction(latent)
            if random.random() < epsilon:
                return random.randint(0, logits.shape[-1] - 1)
            return torch.argmax(logits, dim=-1).item()

    def train(self):
        obs = self.env.reset()[0]
        episode_reward = 0
        total_steps = self.config["total_timesteps"]

        for step in range(total_steps):
            action = self.select_action(obs, epsilon=0.1)
            next_obs, reward, terminated, truncated, _ = self.env.step([action])
            done = terminated[0] or truncated[0]
            self.memory.append((obs, action, reward[0], next_obs[0], done))
            obs = next_obs[0] if not done else self.env.reset()[0]
            episode_reward += reward[0]

            if done:
                print(f"Episode reward: {episode_reward}")
                episode_reward = 0

            if len(self.memory) >= self.batch_size:
                self.learn_from_memory()

    def learn_from_memory(self):
        batch = random.sample(self.memory, self.batch_size)
        obs, act, rew, next_obs, done = zip(*batch)

        obs       = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions   = torch.tensor(act, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards   = torch.tensor(rew, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs  = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        dones     = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # MuESLI forward pass
        latent     = self.representation(obs)
        pred_policy, pred_value = self.prediction(latent)

        next_latent = self.representation(next_obs)
        predicted_next_latent, predicted_reward = self.dynamics(latent, actions)

        # Losses
        value_loss = F.mse_loss(pred_value.squeeze(), rewards.squeeze())
        reward_loss = F.mse_loss(predicted_reward.squeeze(), rewards.squeeze())
        policy_loss = F.cross_entropy(pred_policy, actions.squeeze())
        consistency_loss = F.mse_loss(predicted_next_latent, next_latent.detach())

        loss = value_loss + reward_loss + policy_loss + consistency_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
