import os
import argparse
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

import gymnasium as gym
import torch
from utils import create_env, evaluate_agent, record_video, LRSchedulerCallback

# Check CUDA availability
device = "cuda:0"
random_seed = 42
batch_size = 128
exploration_settings = {
    "A": {"initial": 1.0, "final": 0.05, "fraction": 0.1}, # default settings, fast exploitation 
    "B": {"initial": 1.0, "final": 0.01, "fraction": 0.5}, # longer exploration
    "C": {"initial": 1.0, "final": 0.1, "fraction": 0.3}, # medium exploration
}
exp_name = "PPO_C_exploration"
save_dir = "results"

print(f"Using device: {device}")

def train_agent(env_name: str, total_timesteps: int = 100_000, lr: float = 3e-4):
    """Train an agent on a classic control environment using PPO with CUDA support."""
    
    # Create directory for saving results
    os.makedirs(save_dir, exist_ok=True)
    
    # Create environment
    env = create_env(env_name)
    
    # Create evaluation environment
    eval_env = create_env(env_name)
    
    # Create the model with CUDA support
    if env_name.startswith("ALE/"):
        model = DQN(
            "CnnPolicy",
            env,
            verbose=1,
            seed=random_seed,
            learning_rate=lr,
            tensorboard_log=f"{save_dir}/{env_name}/logs",
            exploration_initial_eps=exploration_settings["C"]["initial"],
            exploration_final_eps=exploration_settings["C"]["final"],
            exploration_fraction=exploration_settings["C"]["fraction"],
            # tau=0.005,
            batch_size=batch_size,
            device=device
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=random_seed,
            learning_rate=lr,
            batch_size=batch_size,
            tensorboard_log=f"{save_dir}/{env_name}/logs",
            device=device
        )
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/{env_name}/weights/{exp_name}",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    lr_callback = LRSchedulerCallback(total_timesteps=total_timesteps)
    callback = CallbackList([eval_callback, lr_callback])

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
        tb_log_name=exp_name
    )
    
    # Save the final model
    model.save(f"{save_dir}/{env_name}/weights/{exp_name}/final_model")
    
    # Close environments
    env.close()
    eval_env.close()

def test_agent(env_name: str):
    model_path = f"{save_dir}/{env_name}/weights/{exp_name}/final_model"
    env = gym.make(env_name)

    if env_name.startswith("ALE/"):
        model = DQN.load(model_path, device=device)
    else:
        model = PPO.load(model_path, device=device)
        
    mean_reward, std_reward = evaluate_agent(env, model)
    print(f"Final mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    video_path = f"videos/{env_name}"
    record_video(env, model, video_path)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train an agent on a classic control environment")
    parser.add_argument("--env", type=str, default="CartPole-v1", 
                        help="Environment name (default: CartPole-v1)")
    parser.add_argument("--timesteps", type=int, default=100_000, 
                        help="Total timesteps for training (default: 100,000)")
    parser.add_argument("--lr", type=float, default=3e-4, 
                        help="Learning rate (default: 0.0003)")
    parser.add_argument("--test", action="store_true",
                        help="Test the agent (default: False)")
    args = parser.parse_args()
    
    """ Example environments: """
    # CartPole-v1
    # MountainCar-v0
    # Acrobot-v1
    # Pendulum-v1
    # LunarLander-v3 (requires Box2D)

    """ Atari environments: """
    # ALE/Breakout-v5
    # ALE/SpaceInvaders-v5
    # ALE/MsPacman-v5
    # ALE/MontezumaRevenge-v5
    # ALE/Seaquest-v5
    
    if not args.test:
        print(f"Training on environment: {args.env}")
        train_agent(
            env_name=args.env,
            total_timesteps=args.timesteps,
            lr=args.lr
        ) 
    test_agent(
        env_name=args.env,
    )