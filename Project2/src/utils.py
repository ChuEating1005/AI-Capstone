import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional
import os
import ale_py
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import update_learning_rate

# Explicitly register ALE environments
try:
    gym.register_envs(ale_py)
    print("ALE environments registered successfully")
except Exception as e:
    print(f"Warning: Failed to register ALE environments: {e}")  

class LRSchedulerCallback(BaseCallback):
    def lr_schedule(self):
        initial_lr = 3e-4
        min_lr = 1e-5
        return max(min_lr, initial_lr - (initial_lr - min_lr) * (self.num_timesteps / self.total_timesteps))
    
    def __init__(self, verbose=0, total_timesteps=1_000_000):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        
    def _on_step(self):
        new_lr = self.lr_schedule()
        update_learning_rate(self.model.policy.optimizer, new_lr)
        return True



def create_env(env_name) -> gym.Env:
    """Create and return a Gymnasium environment."""
    # Special configuration for Atari environments
    if env_name.startswith("ALE/"):
        env = gym.make(
            env_name, 
            frameskip=4,  # Use a standard frame skip
            repeat_action_probability=0.25,  # Sticky actions (v5 standard)
            full_action_space=False  # Use minimal action space
        )
        env = Monitor(env)
        dummy_env = DummyVecEnv([lambda: env])
        return VecTransposeImage(dummy_env)
    else:
        env = gym.make(env_name)
        env = Monitor(env)
        return DummyVecEnv([lambda: env])

def evaluate_agent(env, model, n_eval_episodes=10):
    """Evaluate a trained agent."""
    episode_rewards = []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        # Handle both old and new Gymnasium API
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        episode_rewards.append(episode_reward)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward

def record_video(env: gym.Env, agent: Any, video_path: str, fps: int = 30):
    """Record a video of the agent playing."""
    # For Atari games, use a proper rendering
    env = gym.make(
        env.unwrapped.spec.id, 
        render_mode="rgb_array"
    )
    
    # Setup video recording
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=os.path.dirname(video_path),
        name_prefix=os.path.basename(video_path).split('.')[0],
        episode_trigger=lambda x: True,  # Record every episode
        video_length=0,  # Record entire episodes
        disable_logger=True
    )
    
    # Play one episode
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.predict(obs, deterministic=True)[0]
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    env.close()
    print(f"Episode finished with reward: {total_reward}")
    return total_reward 