import sys, os, yaml, time
import numpy as np
import torch
import gymnasium as gym

# Import custom modules for environment and models
from models import BasicLSTMNet, SpatioTemporalRLNNet, TransformerNet, DuelingTransformerNet, SpatioTempDuelingTransformerNet
# gym_interface is used for reference; we won't vectorize env for rendering
# (No need to import F110EnvWrapper since we'll create env directly for rendering)

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_agent.py <path_to_results_dir>")
        sys.exit(1)
    results_dir = sys.argv[1]
    config_path = os.path.join(results_dir, "config.yaml")
    model_path = os.path.join(results_dir, "best_agent.pt")
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        sys.exit(1)
    # Load configuration
    with open(config_path, 'r') as cf:
        config = yaml.safe_load(cf)
    algorithm = config.get("algorithm", "ppo").lower()
    model_type = config.get("model_type", "basic_lstm").lower()
    # Ensure single-environment for visualization
    config['n_envs'] = 1
    # Disable randomness and noise for clarity in visualization
    config['domain_randomization'] = False
    config['sensor_noise'] = {'lidar': 0.0, 'speed': 0.0}
    # Prepare the environment configuration for gym.make
    env_id = config.get("env_id", "f1tenth_gym:f1tenth-v0")
    map_path = config.get("map_path", None)
    # If a map path is provided, use it; otherwise environment default map is used
    env_config = {
        "map": map_path if map_path is not None else "default",
        "num_agents": 1,
        "model": "dynamic_ST",
        "enable_rendering": 1,
        "enable_scan": 1
    }
    # Choose render mode ('human' for real-time, 'unlimited' for as-fast-as-possible)
    render_mode = "human"

    # just before gym.make(...)
    # first try ST; if it bombs, fall back to plain dynamic
    try:
        env = gym.make(
            env_id,
            map=map_path if map_path is not None else "default",
            num_agents=1,
            model="dynamic_ST",
            enable_rendering=True,
            enable_scan=True,
            render_mode=render_mode,
        )
    except ValueError as e:
        if "dynamic_ST" in str(e):
            print("dynamic_ST not found – falling back to 'dynamic'")
            env = gym.make(
                env_id,
                map=map_path if map_path is not None else "default",
                num_agents=1,
                model="dynamic",
                enable_rendering=True,
                enable_scan=True,
                render_mode=render_mode,
            )
        else:
            raise

    #env = gym.make(env_id, config=env_config, render_mode=render_mode)


    # Determine observation and action dimensions
    # If using F110EnvWrapper, observation_space would be processed shape, but we created env directly.
    # We can infer obs_dim and action_dim from the environment or config.
    # Underlying env observation is a dict; our networks expect flat vectors as trained.
    # We'll extract and process observations similarly to F110EnvWrapper.
    # Determine obs_dim (speed + LiDAR points as in training)
    lidar_enabled = config.get("lidar", {}).get("enabled", True)
    lidar_downsample = config.get("lidar", {}).get("downsample", False)
    full_lidar_dim = 1080
    lidar_dim = 108 if (lidar_enabled and lidar_downsample) else (full_lidar_dim if lidar_enabled else 0)
    include_vel = config.get("include_velocity_in_obs", True)
    state_dim = 1 if include_vel else 0
    obs_dim = state_dim + lidar_dim
    # Determine action dimension or number of actions
    if algorithm == "dqn":
        # Discrete actions (e.g., 5 possible actions)
        num_actions = 5  # default number of discrete actions used in training:contentReference[oaicite:8]{index=8}
        action_dim = num_actions
    else:
        # Continuous action (2-dimensional: steering & speed)
        # If continuous, env.action_space is Box(low=[-..., 0], high=[..., max_speed])
        # We'll treat action_dim as the length of the action vector.
        if hasattr(env.action_space, 'shape'):
            action_dim = int(np.prod(env.action_space.shape))
        else:
            action_dim = env.action_space.n

    # Initialize the appropriate model architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if algorithm == "dqn":
        # For DQN agents, use a dueling architecture (as used in training)
        if model_type == "spatiotemp_dueling_transformer":
            # Spatio-temporal dueling transformer (expects seq_len and num_ranges from config)
            seq_len = config.get("seq_len", 4)
            num_ranges = config.get("num_ranges", 1080)
            model = SpatioTempDuelingTransformerNet(seq_len, num_ranges, action_dim)
        else:
            # Default to dueling transformer network for DQN
            model = DuelingTransformerNet(obs_dim, action_dim)
    else:
        # PPO or other continuous control algorithm (actor-critic network outputs mean action & value)
        if model_type == "basic_lstm":
            model = BasicLSTMNet(obs_dim, action_dim)
        elif model_type == "spatiotemporal_rln":
            model = SpatioTemporalRLNNet(obs_dim, action_dim)
        elif model_type == "transformer":
            model = TransformerNet(obs_dim, action_dim)
        elif model_type == "dueling_transformer":
            # Treat dueling transformer as a standard Transformer for actor-critic
            model = TransformerNet(obs_dim, action_dim)
        elif model_type == "spatiotemp_dueling_transformer":
            # Not originally designed for PPO; fallback to a basic network
            model = BasicLSTMNet(obs_dim, action_dim)
        else:
            model = BasicLSTMNet(obs_dim, action_dim)
    # Load the trained weights into the model
    model_file = model_path
    if algorithm == "dqn" and not os.path.exists(model_file):
        # If best_agent.pt not found (DQN case), try alternate filename (e.g. q_net_final.pth)
        alt_path = os.path.join(results_dir, "q_net_final.pth")
        if os.path.exists(alt_path):
            model_file = alt_path

    ########################################################

    ckpt = torch.load(model_file, map_location=device)

    # SKRL saves {"policy": state_dict, "value": ..., ...}
    if isinstance(ckpt, dict) and "policy" in ckpt:
        policy_sd = ckpt["policy"]

        # If weights are prefixed with "base.", strip that so they
        # match your backbone network's layer names
        trimmed_sd = {k.replace("base.", ""): v for k, v in policy_sd.items()
                    if k.startswith("base.")} or policy_sd

        # load into backbone; ignore extras that don't exist in the viz model
        model.load_state_dict(trimmed_sd, strict=False)
    else:
        # fallback: assume it's already a plain state-dict
        model.load_state_dict(ckpt)

    model.to(device).eval()

    #######################################

    #model.load_state_dict(torch.load(model_file, map_location=device))
    #model.to(device)
    #model.eval()

    # Precompute discrete action mapping for DQN (steering, velocity pairs)
    discrete_actions = None
    if algorithm == "dqn":
        discrete_actions = np.array([
            [-0.4, 5.0],   # steer left, medium speed
            [ 0.0, 5.0],   # go straight, medium speed
            [ 0.4, 5.0],   # steer right, medium speed
            [ 0.0, 2.0],   # go straight, slow speed
            [ 0.0, 8.0]    # go straight, fast speed
        ], dtype=np.float32)  # :contentReference[oaicite:11]{index=11}
    
    print("Loaded model:", model_type, "| Algorithm:", algorithm)
    print("Starting simulation... (Press Ctrl+C to quit)")

    episode = 0
    try:
        while True:  # loop over episodes
            episode += 1
            # Reset environment at the start of each episode
            obs_dict, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
            # Reset episode metrics
            step_count = 0
            cum_reward = 0.0
            start_time = time.time()
            # Simulation (simulated) time counter
            sim_time = 0.0
            done = False
            # For recurrent models (LSTM), initialize hidden states
            lstm_hidden = None
            lstm_cell = None
            if isinstance(model, BasicLSTMNet) or isinstance(model, SpatioTemporalRLNNet):
                # Prepare zero hidden and cell states for LSTM (shape: (1, batch=1, hidden_dim))
                h_dim = model.hidden_dim if hasattr(model, "hidden_dim") else 128
                lstm_hidden = torch.zeros((1, 1, h_dim)).to(device)
                lstm_cell = torch.zeros((1, 1, h_dim)).to(device)
            # Run one episode
            while not done:
                # Process observation: flatten into state vector (speed + LiDAR) as per training
                obs_vec = []
                if include_vel:
                    # Forward speed is at index 3 of 'std_state':contentReference[oaicite:12]{index=12}
                    speed = float(obs_dict['agent_0']['std_state'][3])
                    obs_vec.append(speed)
                if lidar_enabled:
                    # Extract LiDAR scan array from observation:contentReference[oaicite:13]{index=13}
                    lidar_scan = None
                    for key in ("scans", "lidar", "laser_scan", "ranges"):
                        if key in obs_dict:
                            val = obs_dict[key]
                            lidar_scan = val[0] if hasattr(val, "__len__") else val
                            break
                    if lidar_scan is not None:
                        if lidar_downsample:
                            lidar_scan = lidar_scan[::10]  # downsample from 1080 to 108 beams
                        obs_vec.extend(lidar_scan.astype(np.float32).tolist())
                # Pad or trim obs_vec to expected obs_dim (safety check)
                if len(obs_vec) < obs_dim:
                    obs_vec.extend([0.0] * (obs_dim - len(obs_vec)))
                elif len(obs_vec) > obs_dim:
                    obs_vec = obs_vec[:obs_dim]
                obs_array = np.array(obs_vec, dtype=np.float32)
                # Model inference
                obs_tensor = torch.from_numpy(obs_array).to(device).unsqueeze(0)  # shape (1, obs_dim)
                if isinstance(model, BasicLSTMNet) or isinstance(model, SpatioTemporalRLNNet):
                    # For LSTM-based models, input shape should be (seq_len, batch, obs_dim)
                    # Here we feed one timestep at a time. Use hidden state if maintaining LSTM state.
                    inp = obs_tensor.unsqueeze(0)  # (1, 1, obs_dim)
                    # Get output (and optionally new hidden state if we modify model to return it)
                    action_mean, _ = model(inp)  # Note: this resets LSTM hidden each forward by default
                else:
                    # Feed-forward models (or treating each step independently)
                    output = model(obs_tensor)
                    if isinstance(output, tuple) or isinstance(output, list):
                        action_mean = output[0]  # for actor-critic networks
                    else:
                        action_mean = output  # for Q-network, this *is* Q-values
                action = None
                if algorithm == "dqn":
                    # For DQN: output is Q-values for each discrete action
                    q_values = action_mean.detach().cpu().numpy().flatten()
                    act_index = int(np.argmax(q_values))
                    # Map discrete index to continuous action values:contentReference[oaicite:14]{index=14}
                    action = discrete_actions[act_index]
                else:
                    # For PPO (continuous): output `action_mean` is the mean action; use it directly
                    action = action_mean.detach().cpu().numpy().flatten()
                    # If action space is bounded, we may want to clamp to valid range (optional)

                if algorithm == "dqn":
                    # lookup already returns (2,)
                    action_vec = action                     # ndarray (2,)
                else:
                    # PPO: may be (1,) or (2,)
                    action_vec = action

                    if action_vec.size == 1:                # network only predicts steer
                        action_vec = np.array([action_vec[0], 5.0], dtype=np.float32)  # add 5 m/s

                # single-agent  →  (2,); multi-agent → (n, 2)
                if env.unwrapped.num_agents == 1:
                    action_to_env = action_vec.astype(np.float32)          # shape (2,)
                else:
                    action_to_env = action_vec.reshape(env.unwrapped.num_agents, 2)

                # Step the environment
                obs_step = env.step(action_to_env)


                # Ensure shape (num_agents, 2) for f1tenth-gym
                #action = np.asarray(action, dtype=np.float32).reshape(1, -1)

                # Step the environment with the selected action
                #obs_step = env.step(action)



                # Gymnasium's step can return (obs, reward, done, truncated, info) or (obs, reward, done, info)
                if isinstance(obs_step, tuple) and len(obs_step) == 5:
                    next_obs_dict, reward, terminated, truncated, info = obs_step
                    done = terminated or truncated
                else:
                    # Older gym API compatibility
                    next_obs_dict, reward, done, info = obs_step
                # Update simulation time (assuming simulator time step ~0.01s)
                # If available, use environment's time_step attribute; else default 0.01
                dt = getattr(env.unwrapped, "time_step", 0.01)
                sim_time += dt
                step_count += 1
                cum_reward += float(reward)
                # Compute instantaneous FPS (for performance monitoring)
                # (We compute based on wall-clock time between steps to display actual frame rate)
                # Use a short moving window for smoothing
                if 'fps_times' not in locals():
                    fps_times = []
                fps_times.append(time.time())
                if len(fps_times) > 30:
                    fps_times.pop(0)
                fps = 0.0
                if len(fps_times) >= 2:
                    fps = (len(fps_times)-1) / (fps_times[-1] - fps_times[0] + 1e-6)
                # Update overlay text if possible (using environment's renderer text objects, if present)
                if hasattr(env, "renderer"):
                    try:
                        # Update an FPS display if exists
                        if hasattr(env.renderer, "fps_display") and hasattr(env.renderer.fps_display, "label"):
                            env.renderer.fps_display.label.text = f"FPS: {fps:.1f}"
                        # Update score or status label if exists
                        if hasattr(env.renderer, "score_label"):
                            env.renderer.score_label.text = f"Step: {step_count}   Time: {sim_time:.2f}s   Reward: {cum_reward:.2f}"
                    except Exception:
                        pass
                # Render the frame (this updates the GUI)
                env.render()
                # Prepare for next loop iteration
                obs_dict = next_obs_dict
            # Episode ended, print summary
            outcome = "finished lap" if step_count >= config.get("max_episode_steps", np.inf) or info.get('finished', False) else "crashed"
            print(f"Episode {episode} ended - Steps: {step_count}, Lap Time: {sim_time:.2f}s, Cumulative Reward: {cum_reward:.2f} ({outcome})")
    except KeyboardInterrupt:
        print("Visualization stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
