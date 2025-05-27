import os
import numpy as np
import gym
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import pandas as pd

class CustomCallback(BaseCallback):
    """
    Custom callback for saving models and tracking metrics during training.
    """
    def __init__(self, check_freq=1000, save_path=None, verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.rewards = []
        self.portfolio_values = []
        self.drawdowns = []
        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            if self.save_path is not None:
                self.model.save(f"{self.save_path}/model_{self.n_calls}")
                
            # Record metrics from the environment
            info = self.locals['infos'][0]  # Get info from first environment
            if 'portfolio_value' in info:
                self.portfolio_values.append(info['portfolio_value'])
            if 'drawdown' in info:
                self.drawdowns.append(info['drawdown'])
                
            # Calculate episodic reward
            if self.locals['dones'][0]:
                ep_reward = np.sum(self.locals['rewards'])
                self.rewards.append(ep_reward)
                
                if self.verbose > 0:
                    print(f"Step: {self.n_calls}, Reward: {ep_reward:.2f}")
                    
        return True

class PPOAgent:
    """
    A PPO agent wrapper using Stable-Baselines3.
    """
    def __init__(self, env, model_kwargs=None):
        """
        Initialize the PPO agent.
        
        Args:
            env: The gym environment
            model_kwargs: Additional kwargs for the PPO model
        """
        # Create logging directory
        log_dir = "logs/ppo_agent/"
        os.makedirs(log_dir, exist_ok=True)
        
        # Check if environment is a gym/gymnasium environment
        # If not, we'll use it directly without Monitor wrapping
        try:
            # Skip the Monitor wrapper since it's causing issues
            # Just use the environment directly
            self.env = env
            
            # Wrap with DummyVecEnv as PPO expects a vectorized environment
            self.vec_env = DummyVecEnv([lambda: self.env])
            
        except Exception as e:
            print(f"Warning: Couldn't wrap environment with Monitor: {e}")
            print("Using environment directly without Monitor wrapper.")
            self.env = env
            self.vec_env = DummyVecEnv([lambda: self.env])
        
        # Set default model parameters
        if model_kwargs is None:
            model_kwargs = {
                "policy": "MlpPolicy",
                "learning_rate": 1e-4,
                "gamma": 0.99,
                "verbose": 1
            }
        
        # Ensure policy is specified
        if "policy" not in model_kwargs:
            model_kwargs["policy"] = "MlpPolicy"
            
        # Initialize the PPO model
        self.model = PPO(env=self.vec_env, **model_kwargs)
        
        # Initialize callback
        self.callback = None
        
    def train(self, total_timesteps=100000, save_path=None, check_freq=10000):
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Number of timesteps to train for
            save_path: Path to save model checkpoints
            check_freq: Frequency of saving checkpoints
            
        Returns:
            Training metrics from the callback
        """
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            
        # Create callback
        self.callback = CustomCallback(check_freq=check_freq, save_path=save_path)
        
        # Train the model
        self.model.learn(total_timesteps=total_timesteps, callback=self.callback)
        
        # Return training metrics
        return {
            'rewards': self.callback.rewards,
            'portfolio_values': self.callback.portfolio_values,
            'drawdowns': self.callback.drawdowns
        }
    
    def predict(self, observation, deterministic=True):
        """
        Predict the best action for a given observation.
        
        Args:
            observation: The current observation from the environment
            deterministic: Whether to use deterministic actions or sample from distribution
            
        Returns:
            action: The selected action
            _: The state (unused)
        """
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, filepath):
        """
        Save the PPO model.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load a saved PPO model.
        
        Args:
            filepath: Path to load the model from
        """
        self.model = PPO.load(filepath, env=self.vec_env)
        print(f"Model loaded from {filepath}")
    
    def plot_training_metrics(self, metrics=None, filepath=None):
        """
        Plot training metrics.
        
        Args:
            metrics: Dictionary of training metrics (if None, use callback metrics)
            filepath: Optional filepath to save the plot
        """
        if metrics is None and self.callback is None:
            print("No training metrics available to plot")
            return
            
        if metrics is None:
            metrics = {
                'rewards': self.callback.rewards,
                'portfolio_values': self.callback.portfolio_values,
                'drawdowns': self.callback.drawdowns
            }
            
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
        
        # Plot rewards
        if metrics['rewards']:
            ax1.plot(metrics['rewards'])
            ax1.set_title('Episode Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.grid(True)
        
        # Plot portfolio values
        if metrics['portfolio_values']:
            ax2.plot(metrics['portfolio_values'])
            ax2.set_title('Portfolio Value')
            ax2.set_xlabel('Check Frequency')
            ax2.set_ylabel('Value ($)')
            ax2.grid(True)
        
        # Plot drawdowns
        if metrics['drawdowns']:
            ax3.plot([d*100 for d in metrics['drawdowns']])
            ax3.set_title('Drawdown')
            ax3.set_xlabel('Check Frequency')
            ax3.set_ylabel('Drawdown (%)')
            ax3.grid(True)
        
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath)
            print(f"Training metrics saved to {filepath}")
        
        plt.show()
