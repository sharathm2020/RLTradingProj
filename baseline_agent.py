from env.trading_env import TradingEnv
from utils.helpers import load_stock_data
import numpy as np
import matplotlib.pyplot as plt
import os

def run_baseline_agent(ticker="AAPL", start="2020-01-01", end="2023-01-01", episodes=1, max_trades=50):
    """
    Run a baseline random agent and save performance results.
    
    Args:
        ticker: Stock ticker symbol
        start: Start date for data
        end: End date for data
        episodes: Number of episodes to run
        max_trades: Maximum number of trades to allow before penalty
    """
    print(f"Running baseline agent for {ticker} from {start} to {end}")
    
    # Create results directory
    results_dir = f"results/baseline_{ticker}_{start}_{end}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    df = load_stock_data(ticker=ticker, start=start, end=end)
    print(f"Loaded {len(df)} data points")
    
    # Create environment
    env = TradingEnv(df, initial_balance=10000, max_trades=max_trades)
    
    episode_rewards = []
    episode_portfolio_values = []
    episode_drawdowns = []
    
    # Run episodes
    for episode in range(episodes):
        print(f"\nEpisode {episode+1}/{episodes}")
        
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Run steps until episode is done
        while not done:
            # Random action
            action = env.action_space.sample()
            
            # Take step
            obs, reward, done, info = env.step(action)
            reward = reward.item() if hasattr(reward, "item") else float(reward)
            
            total_reward += reward
            steps += 1
            
            if steps % 50 == 0:
                print(f"Step {steps}, Portfolio Value: ${info['portfolio_value']:.2f}, " 
                      f"Drawdown: {info['drawdown']*100:.2f}%")
        
        # Episode summary
        print(f"Episode {episode+1} completed in {steps} steps")
        print(f"Final portfolio value: ${env.portfolio_values[-1]:.2f}")
        print(f"Return: {((env.portfolio_values[-1] / env.initial_balance) - 1) * 100:.2f}%")
        print(f"Max drawdown: {env.max_drawdown * 100:.2f}%")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Number of trades: {env.trade_count}")
        
        # Collect episode stats
        episode_rewards.append(total_reward)
        episode_portfolio_values.append(env.portfolio_values)
        episode_drawdowns.append(env.max_drawdown)
        
        # Save episode results
        env.save_results(f"{results_dir}/episode_{episode+1}")
    
    # Plot and save episode rewards
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodes+1), episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(f"{results_dir}/episode_rewards.png")
    
    # Plot and save max drawdowns
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, episodes+1), [d*100 for d in episode_drawdowns])
    plt.title('Max Drawdown per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Max Drawdown (%)')
    plt.grid(True)
    plt.savefig(f"{results_dir}/episode_drawdowns.png")
    
    print(f"\nBaseline experiments completed and saved to {results_dir}")
    return episode_rewards, episode_portfolio_values, episode_drawdowns

if __name__ == "__main__":
    # Run baseline agent with default parameters
    run_baseline_agent(episodes=3) 