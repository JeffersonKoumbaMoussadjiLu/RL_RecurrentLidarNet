import os
import numpy as np
import gymnasium as gym
from f1tenth_gym.f110_gym.envs.f110_env import F110Env

gym.register(
    id="f1tenth-v0",
    entry_point="f1tenth_gym.f110_gym.envs.f110_env:F110Env",
)

from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from skrl.envs.wrappers.torch import wrap_env
# Custom environment wrapper for F1TENTH gym to handle observation processing, noise, and reward shaping.
class F110EnvWrapper(gym.Env):
    def __init__(self, config, seed=0):
        super().__init__()
        self.config = config
        self.seed   = seed

        self.discrete_actions = None
        env_kwargs = {
            "seed":       seed,
            "model":      "dynamic_ST",
            "num_agents": 1
        }

        # if the user specified a map_path, strip its ".yaml" so F110Env can find both .yaml and .png
        map_path = config.get("map_path", None)
        if map_path is not None:
            base, ext = os.path.splitext(map_path)
            env_kwargs["map"] = base if ext.lower() == ".yaml" else map_path

        # —— vehicle parameters ——  
        # must include every entry F110Env expects, so start from its defaults
        default_params = {
            'mu': 1.0489,   'C_Sf': 4.718,   'C_Sr': 5.4562,
            'lf': 0.15875,  'lr': 0.17145,  'h': 0.074,
            'm': 3.74,      'I': 0.04712,
            's_min': -0.4189, 's_max': 0.4189,
            'sv_min': -3.2,   'sv_max': 3.2,
            'v_switch': 7.319, 'a_max': 9.51,
            'v_min': -5.0,     'v_max': 20.0,
            'width': 0.31,     'length': 0.58
        }

        # only patch in the randomized bits
        if config.get("domain_randomization", False):
            rng = np.random.RandomState(seed)
            params = default_params.copy()
            params.update({
                'mu':   rng.uniform(0.8, 1.2),
                'C_Sf': rng.uniform(4.0, 5.5),
                'C_Sr': rng.uniform(4.0, 5.5),
                'm':    rng.uniform(3.0, 4.5),
                'I':    rng.uniform(0.04, 0.05)
            })
            env_kwargs["params"] = params
        # if not randomizing, leave out "params" entirely – F110Env.__init__ will fall back to its own defaults

        # now import & instantiate the real env
        from f1tenth_gym.f110_gym.envs.f110_env import F110Env
        self.env = F110Env(**env_kwargs)


        self._max_episode_steps = config.get("max_episode_steps", None)
        self.current_step       = 0

        lidar_enabled    = config["lidar"]["enabled"]
        lidar_downsample = config["lidar"]["downsample"]
        full_dim         = 1080
        lidar_dim        = 108 if (lidar_enabled and lidar_downsample) else (full_dim if lidar_enabled else 0)
        state_dim        = 1 if config.get("include_velocity_in_obs", True) else 0
        obs_dim          = state_dim + lidar_dim
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # for BOTH DQN and Muesli we use the same discrete action set
        if config["algorithm"].lower() in ["dqn", "muesli"]:
            self.discrete_actions = np.array([
                [-0.4, 5.0],
                [ 0.0, 5.0],
                [ 0.4, 5.0],
                [ 0.0, 2.0],
                [ 0.0, 8.0]
            ], dtype=np.float32)
            self.action_space = gym.spaces.Discrete(len(self.discrete_actions))
        else:
            # continuous actions for PPO/A2C
            sim_params = self.env.sim.params
            low  = np.array([sim_params['s_min'], sim_params['v_min']], dtype=np.float32)
            high = np.array([sim_params['s_max'], sim_params['v_max']], dtype=np.float32)
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)


        noise_cfg = config.get("sensor_noise", {})
        self.lidar_noise_std = noise_cfg.get("lidar", 0.0)
        self.speed_noise_std = noise_cfg.get("speed", 0.0)

        self.last_speed             = 0.0
        self.last_steer             = 0.0
        self.total_abs_speed        = 0.0
        self.total_abs_steer_change = 0.0


    def _extract_speed(self, obs: dict) -> float:
        """
        Pull out the ego vehicle’s forward speed (m/s) from the raw obs dict.

        Tries, in order:
        1. The new F110Env format: `linear_vels_x` + `linear_vels_y` + `ego_idx`
        2. Legacy flat-array style: `std_state`
        3. Legacy nested style: `obs['agent_0']['std_state']`
        """
        # 1) New-style: use linear_vels_x & linear_vels_y
        if "linear_vels_x" in obs and "linear_vels_y" in obs:
            vx = np.asarray(obs["linear_vels_x"])
            vy = np.asarray(obs["linear_vels_y"])
            idx = int(obs.get("ego_idx", 0))
            # magnitude of the first (or ego) agent’s velocity
            return float(np.sqrt(vx[idx]**2 + vy[idx]**2))

        # 2) Flat std_state style
        if "std_state" in obs:
            arr = np.asarray(obs["std_state"])
            # shape (num_agents, state_dim) or (state_dim,)
            if arr.ndim == 2:
                return float(arr[0, 3])
            elif arr.ndim == 1:
                return float(arr[3])

        # 3) Nested agent_0 style
        if "agent_0" in obs and isinstance(obs["agent_0"], dict):
            ss = obs["agent_0"].get("std_state")
            if ss is not None and len(ss) > 3:
                return float(ss[3])

        raise RuntimeError(f"Cannot extract speed; obs keys are {list(obs.keys())!r}")




    def _extract_lidar(self, obs):
        """Extract LiDAR scan array from observation dict (handles different keys)."""
        for k in ("scans", "lidar", "laser_scan", "ranges"):
            if k in obs:
                return obs[k][0] if hasattr(obs[k], "__len__") else obs[k]
        return None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the environment and return (observation, info)."""
        # reset our step counter
        self.current_step = 0
        if seed is not None:
            self.seed = seed

        # build default zero poses for all agents
        num_agents = getattr(self.env, "num_agents", 1)
        poses = np.zeros((num_agents, 3), dtype=np.float32)

        # call underlying reset
        result = self.env.reset(poses)

        # unpack either 4-tuple or 5-tuple
        if len(result) == 4:
            obs_dict, reward, done, info = result
        elif len(result) == 5:
            obs_dict, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            raise RuntimeError(f"Unexpected reset return shape: {result}")

        # clear any wrapper metrics
        self.last_speed             = 0.0
        self.last_steer             = 0.0
        self.total_abs_speed        = 0.0
        self.total_abs_steer_change = 0.0

        # process and return only the observation and info
        processed_obs = self._process_obs(obs_dict)
        return processed_obs, info


    def step(self, action):
        """Step the environment with the given action and return processed observation, reward, done flags, and info."""
        self.current_step += 1
        # Map discrete action index to actual action values if using discrete actions
        if self.discrete_actions is not None:
            actual_action = self.discrete_actions[action]
        else:
            actual_action = np.array(action, dtype=np.float32)
            if actual_action.ndim == 1:
                actual_action = actual_action[None, :]  # add batch dimension for single agent

        # Ensure action shape matches number of agents in underlying env
        try:
            n_agents = len(self.env.agents)  # for newer f1tenth_gym versions
        except AttributeError:
            n_agents = getattr(self.env, "num_agents", 1)
        if actual_action.ndim == 1:
            actual_action = np.tile(actual_action, (n_agents, 1))
        elif actual_action.shape[0] != n_agents:
            actual_action = np.tile(actual_action[0], (n_agents, 1))

        # Step the underlying environment
        result = self.env.step(actual_action)
        if len(result) == 5:
            obs_dict, env_reward, terminated, truncated, info = result
        else:
            obs_dict, env_reward, done, info = result
            terminated, truncated = done, False  # ensure terminated/truncated flags

        # Compute shaped reward components
        speed = abs(self._extract_speed(obs_dict))
        steer = float(actual_action[0, 0])  # steering command of first agent
        steering_delta = abs(steer - self.last_steer)
        progress_reward = 0.0  # TODO: (optional track progress if available)
        speed_reward = speed - 1.0  # reward for speed (baseline 1 m/s)
        accel = abs(speed - self.last_speed)
        collision = bool(obs_dict.get("collisions", [False])[0])
        # Combine reward with weights
        weight = self.config["reward_weights"]
        reward = weight.get("progress", 0.0) * progress_reward \
               + weight.get("speed", 0.0) * speed_reward \
               - weight.get("steering_change", 0.0) * steering_delta \
               - weight.get("acceleration", 0.0) * accel \
               - (weight.get("collision", 0.0) * 1.0 if collision else 0.0)

        # Update cumulative metrics and last values
        self.total_abs_speed += speed
        self.total_abs_steer_change += steering_delta
        self.last_speed = speed
        self.last_steer = steer

        processed_obs = self._process_obs(obs_dict)
        # If episode ends, add custom metrics to info for logging
        if terminated or truncated:
            info_episode = info.get("episode", {})
            info_episode["avg_speed"] = (self.total_abs_speed / self.current_step) if self.current_step > 0 else 0.0
            info_episode["avg_steering_change"] = (self.total_abs_steer_change / self.current_step) if self.current_step > 0 else 0.0
            info["episode"] = info_episode

        return processed_obs, reward, terminated, truncated, info


    '''
    def _process_obs(self, obs_dict):
        """Return a flat vector of length self.observation_space.shape[0]."""
        target_len = self.observation_space.shape[0]
        vec        = np.zeros(target_len, dtype=np.float32)   # pre-allocate

        idx = 0
        # ───── speed ───────────────────────────────────────────────
        if self.config.get("include_velocity_in_obs", True):
            speed = self._extract_speed(obs_dict)
            if self.speed_noise_std > 0:
                speed += np.random.normal(0, self.speed_noise_std)
            vec[idx] = speed
            idx += 1

        # ───── LiDAR ───────────────────────────────────────────────
        if self.config["lidar"]["enabled"]:
            scan = self._extract_lidar(obs_dict)
            if scan is not None:
                if self.config["lidar"]["downsample"]:
                    scan = scan[::10]                        # 108 readings if 1080 raw
                if self.lidar_noise_std > 0:
                    scan = np.clip(
                        scan + np.random.normal(0, self.lidar_noise_std, size=scan.shape) * scan,
                        0.0, np.inf
                    )
                scan_len          = min(len(scan), target_len - idx)
                vec[idx: idx+scan_len] = scan[:scan_len]

        return vec
    '''

    
    def _process_obs(self, obs_dict):
        #Process raw observation dict into a flat NumPy array (add noise, downsample LiDAR, etc.).
        obs_vec = []

        # Include velocity (speed) in observation if enabled
        if self.config.get("include_velocity_in_obs", True):
            speed = self._extract_speed(obs_dict)
            if self.speed_noise_std > 0:
                speed += np.random.normal(0, self.speed_noise_std)
            obs_vec.append(speed)

        # Include LiDAR scan if enabled
        lidar_scan = None
        if self.config["lidar"]["enabled"]:
            lidar_scan = self._extract_lidar(obs_dict)
        if lidar_scan is not None:
            # Downsample LiDAR if configured
            if self.config["lidar"]["downsample"]:
                lidar_scan = lidar_scan[::10]
            # Add noise to LiDAR readings (proportional noise)
            if self.lidar_noise_std > 0:
                noise = np.random.normal(0, self.lidar_noise_std, size=lidar_scan.shape)
                lidar_scan = np.clip(lidar_scan + noise * lidar_scan, 0.0, np.inf)
            obs_vec.extend(lidar_scan.astype(np.float32))
        obs_array = np.array(obs_vec, dtype=np.float32)

        ###### SAFETY PAD so len(obs_vec) == self.observation_space.shape[0] ####
        target = self.observation_space.shape[0]
        if len(obs_vec) < target:
            obs_vec.extend([0.0] * (target - len(obs_vec)))      # pad missing slots
        elif len(obs_vec) > target:
            obs_vec = obs_vec[:target]                           # rare: trim extras

        obs_array = np.array(obs_vec, dtype=np.float32)
        return obs_array
    

def make_vector_env(config):
    """Create a vectorized environment with the specified number of parallel F1TENTH environments."""
    num_envs = config.get("n_envs", 1)
    use_subproc = (num_envs > 1 and os.name != 'nt')  # use subprocesses for parallel envs if not on Windows
    def make_env_fn(rank):
        def _init():
            env = F110EnvWrapper(config, seed=config.get("seed", 0) + rank)
            # Wrap each env to record episode statistics (returns and lengths)
            return RecordEpisodeStatistics(env)
        return _init
    if num_envs == 1:
        env = F110EnvWrapper(config, seed=config.get("seed", 0))
        #env = RecordEpisodeStatistics(env)
        env = wrap_env(env)  # wrap single environment for skrl
    else:
        env_fns = [make_env_fn(i) for i in range(num_envs)]
        env = AsyncVectorEnv(env_fns) if use_subproc else SyncVectorEnv(env_fns)
        env = wrap_env(env)
    return env