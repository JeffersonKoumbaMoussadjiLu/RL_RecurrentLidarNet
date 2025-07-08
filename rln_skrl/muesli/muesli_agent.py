import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class MuesliAgent:
    def __init__(self, env, representation, dynamics, prediction, config, device):
        self.env            = env
        self.representation = representation
        self.dynamics       = dynamics
        self.prediction     = prediction
        self.config         = config
        self.device         = device

        self.optimizer = torch.optim.Adam(
            list(self.representation.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.prediction.parameters()),
            lr=config.get("learning_rate", 3e-4)
        )

        self.memory     = deque(maxlen=config.get("buffer_size", 10000))
        self.batch_size = config.get("batch_size", 32)
        self.gamma      = config.get("gamma", 0.99)

    def select_action(self, state, epsilon=0.1):
        # 1) Turn state into a (1, obs_dim) tensor
        if not isinstance(state, torch.Tensor):
            state_t = torch.from_numpy(state).float().to(self.device)
        else:
            state_t = state.to(self.device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)

        # 2) Forward pass to get logits
        with torch.no_grad():
            latent       = self.representation(state_t)      # (1, hidden)
            logits, _    = self.prediction(latent)           # (1, n_actions)

        # 3) ε-greedy (optional)
        if random.random() < epsilon:
            return random.randrange(logits.size(-1))
        # else pick argmax
        return int(torch.argmax(logits, dim=-1).item())
    
    def train(self):
        obs            = self.env.reset()[0]   # initial obs: np.ndarray
        episode_reward = 0

        for step in range(self.config["total_timesteps"]):
            # 1) get a raw int action
            raw_a = self.select_action(obs, epsilon=0.1)  # Python int in [0,4]

            # 2) pack into a (1,) LongTensor on the right device
            a_t = torch.tensor([raw_a], dtype=torch.long, device=self.device)

            # 3) step the env — skrl will turn that into the right numpy index,
            #    your wrapper will map it to the 2-vector
            next_obs, reward, term, trunc, _ = self.env.step(a_t)
            done = bool(term[0] or trunc[0])

            # 4) store the transition
            self.memory.append((
                obs,            # shape (obs_dim,)
                raw_a,          # int
                float(reward[0]),
                next_obs[0],    # shape (obs_dim,)
                done
            ))

            # 5) advance or reset
            obs = self.env.reset()[0] if done else next_obs[0]
            episode_reward += reward[0]
            if done:
                print(f"Episode reward: {episode_reward}")
                episode_reward = 0

            # 6) learn when you have enough
            if len(self.memory) >= self.batch_size:
                self.learn_from_memory()



    def learn_from_memory(self):
        batch = random.sample(self.memory, self.batch_size)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        # 1) build per-sample tensors, squeeze any leading [1, ...]
        obs_list, next_obs_list = [], []
        for o, no in zip(obs_batch, next_obs_batch):
            o_t  = torch.from_numpy(o).float()     if not isinstance(o, torch.Tensor) else o
            no_t = torch.from_numpy(no).float()    if not isinstance(no, torch.Tensor) else no
            if o_t.dim() == 2 and o_t.size(0) == 1:   o_t  = o_t.squeeze(0)
            if no_t.dim() == 2 and no_t.size(0) == 1: no_t = no_t.squeeze(0)
            obs_list.append(o_t)
            next_obs_list.append(no_t)

        # 2) stack into (B, obs_dim)
        obs      = torch.stack(obs_list,      dim=0).to(self.device)
        next_obs = torch.stack(next_obs_list, dim=0).to(self.device)

        # 3) actions, rewards, dones
        actions = torch.tensor(act_batch, dtype=torch.long,    device=self.device).unsqueeze(1)  # (B,1)
        rewards = torch.tensor(rew_batch, dtype=torch.float32, device=self.device).unsqueeze(1)  # (B,1)
        dones   = torch.tensor(done_batch,dtype=torch.float32, device=self.device).unsqueeze(1)  # (B,1)

        # 4) forward through MuESLI
        latent                   = self.representation(obs)             
        pred_policy, pred_value  = self.prediction(latent)            # pred_value: (B, 21)
        next_latent, pred_r_dist = self.dynamics(latent, actions)     # pred_r_dist: (B, 21)

        # 5) collapse distributions to scalars
        value_pred_scalar = pred_value.mean(dim=1, keepdim=True)      # (B,1)
        reward_pred_scalar = pred_r_dist.mean(dim=1, keepdim=True)    # (B,1)

        # 6) compute losses
        value_loss       = F.mse_loss(value_pred_scalar, rewards)     # both (B,1)
        reward_loss      = F.mse_loss(reward_pred_scalar, rewards)    # both (B,1)
        policy_loss      = F.cross_entropy(pred_policy, actions.squeeze())
        consistency_loss = F.mse_loss(next_latent, next_latent.detach())

        loss = value_loss + reward_loss + policy_loss + consistency_loss

        # 7) optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
