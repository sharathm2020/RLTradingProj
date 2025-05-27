import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

class QLearningAgent:
    """
    A Q-Learning agent for the trading environment.
    
    This agent discretizes the continuous state space into buckets and uses
    a Q-table to make trading decisions based on the current state.
    """
    
    def __init__(self, action_space, observation_space, 
                 learning_rate=0.1, discount_factor=0.95, 
                 exploration_rate=1.0, exploration_decay=0.995,
                 min_exploration_rate=0.01):
        """
        Initialize the Q-Learning agent.
        
        Args:
            action_space: The action space from the environment
            observation_space: The observation space from the environment
            learning_rate: Alpha - the learning rate for Q-value updates
            discount_factor: Gamma - the discount factor for future rewards
            exploration_rate: Epsilon - the initial exploration rate (1.0 = 100% random actions)
            exploration_decay: Rate at which exploration decreases over time
            min_exploration_rate: Minimum exploration rate
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Number of bins for each observation dimension
        self.num_bins = 10
        
        # Initialize Q-table as a dictionary with defaultdict (for unseen states)
        self.q_table = defaultdict(lambda: np.array([0.1, 0.1, 0.1]))

        
        # Training stats
        self.training_rewards = []
        self.exploration_rates = []
        self.q_value_means = []
        self.episode_rewards = 0
    
    def discretize_state(self, state):
        """
        Convert a continuous state into a discrete state by binning the values.
        
        Args:
            state: The continuous state from the environment
            
        Returns:
            A tuple representing the discretized state
        """
        # Get the upper and lower bounds of the observation space
        upper_bounds = self.observation_space.high
        lower_bounds = self.observation_space.low
        
        # Handle infinite bounds (replace with large finite values)
        upper_bounds = np.where(upper_bounds == np.inf, 1e10, upper_bounds)
        lower_bounds = np.where(lower_bounds == -np.inf, -1e10, lower_bounds)
        
        # Width of each bin for each dimension
        widths = (upper_bounds - lower_bounds) / self.num_bins
        
        # Discretize each dimension of the state
        discrete_state = []
        for i, (s, l, w) in enumerate(zip(state, lower_bounds, widths)):
            if w == 0:  # Handle zero width (single value dimension)
                discrete_state.append(0)
            else:
                bin_index = min(self.num_bins - 1, max(0, int((s - l) / w)))
                discrete_state.append(bin_index)
        
        # Return as a tuple (hashable for dictionary key)
        return tuple(discrete_state)

    def act(self, state):
        """
        Choose an action based on the current state using epsilon-greedy policy.
        
        Args:
            state: The current observation from the environment
            
        Returns:
            The selected action
        """
        # Discretize the state
        discrete_state = self.discretize_state(state)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.exploration_rate:
            # Exploration: choose a random action
            return self.action_space.sample()
        else:
            # Exploitation: choose the best action from Q-table
            return np.argmax(self.q_table[discrete_state])

    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-values using the Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Discretize states
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Maximum Q-value for the next state
        max_next_q = np.max(self.q_table[discrete_next_state])
        
        # Current Q-value
        current_q = self.q_table[discrete_state][action]
        
        # Q-learning update rule
        if done:
            # For terminal states, there is no future reward
            new_q = current_q + self.learning_rate * (reward - current_q)
        else:
            # For non-terminal states, consider future rewards
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Update Q-table
        self.q_table[discrete_state][action] = new_q
        
        # Accumulate reward for this episode
        self.episode_rewards += reward
        
        # Decay exploration rate and update stats if episode ended
        if done:
            self.exploration_rate = max(
                self.min_exploration_rate,
                self.exploration_rate * self.exploration_decay
            )
            
            # Store training statistics
            self.training_rewards.append(self.episode_rewards)
            self.exploration_rates.append(self.exploration_rate)
            
            # Calculate mean Q-value (a measure of learning progress)
            q_values = [np.max(q) for q in self.q_table.values()]
            self.q_value_means.append(np.mean(q_values) if q_values else 0)
            
            # Reset episode rewards for next episode
            self.episode_rewards = 0
    
    def save(self, filepath):
        """
        Save the agent's Q-table and parameters to a file.
        
        Args:
            filepath: Path to save the model
        """
        data = {
            'q_table': dict(self.q_table),  # Convert defaultdict to dict for serialization
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate,
            'exploration_decay': self.exploration_decay,
            'min_exploration_rate': self.min_exploration_rate,
            'num_bins': self.num_bins,
            'training_rewards': self.training_rewards,
            'exploration_rates': self.exploration_rates,
            'q_value_means': self.q_value_means
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load the agent's Q-table and parameters from a file.
        
        Args:
            filepath: Path to load the model from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Load parameters
        self.learning_rate = data['learning_rate']
        self.discount_factor = data['discount_factor']
        self.exploration_rate = data['exploration_rate']
        self.exploration_decay = data['exploration_decay']
        self.min_exploration_rate = data['min_exploration_rate']
        self.num_bins = data['num_bins']
        
        # Convert dict back to defaultdict for Q-table
        self.q_table = defaultdict(lambda: np.zeros(self.action_space.n))
        for state, actions in data['q_table'].items():
            self.q_table[state] = actions
        
        # Load training statistics if available
        if 'training_rewards' in data:
            self.training_rewards = data['training_rewards']
        if 'exploration_rates' in data:
            self.exploration_rates = data['exploration_rates']
        if 'q_value_means' in data:
            self.q_value_means = data['q_value_means']
        
        print(f"Model loaded from {filepath}")
    
    def plot_training_stats(self, filepath=None):
        """
        Plot training statistics (rewards, exploration rate, Q-values).
        
        Args:
            filepath: If provided, save the plot to this path
        """
        if not self.training_rewards:
            print("No training data available to plot")
            return
            
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Plot rewards
        ax1.plot(self.training_rewards)
        ax1.set_title('Training Rewards per Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
        
        # Plot exploration rate
        ax2.plot(self.exploration_rates)
        ax2.set_title('Exploration Rate')
        ax2.set_ylabel('Epsilon')
        ax2.grid(True)
        
        # Plot mean Q-values
        ax3.plot(self.q_value_means)
        ax3.set_title('Mean Q-Value')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Mean Q-Value')
        ax3.grid(True)
        
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath)
            print(f"Training stats saved to {filepath}")
        
        plt.show()
