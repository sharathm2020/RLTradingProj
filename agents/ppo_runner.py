from env.trading_env import TradingEnv
from utils.helpers import load_stock_data
from agents.ppo_agent import PPOAgent, CustomCallback
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

def run_ppo_agent(ticker="AAPL", 
                 start="2020-01-01", 
                 end="2023-01-01", 
                 total_timesteps=100000,
                 max_trades=50,
                 learning_rate=0.0001,
                 gamma=0.95,
                 load_model=None,
                 eval_only=False):
    """
    Run a PPO agent for trading.
    
    Args:
        ticker: Stock ticker symbol
        start: Start date for data
        end: End date for data
        total_timesteps: Number of timesteps to train
        max_trades: Maximum number of trades to allow before penalty
        learning_rate: Learning rate for PPO
        gamma: Discount factor for future rewards
        load_model: Path to pre-trained model (if None, train new model)
        eval_only: If True, only evaluate (no training)
    """
    print(f"Running PPO agent for {ticker} from {start} to {end}")
    
    # Create results directory
    results_dir = f"results/ppo_{ticker}_{start}_{end}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    df = load_stock_data(ticker=ticker, start=start, end=end)
    print(f"Loaded {len(df)} data points")
    
    # Normalize the data
    def normalize_data(data):
        # Create a copy to avoid modifying the original
        data = data.copy()
        
        # Calculate rolling statistics using Series
        rolling_mean = data['Close'].rolling(window=20).mean()
        rolling_std = data['Close'].rolling(window=20).std()
        
        # Normalize price using Series operations
        data.loc[:, 'Normalized_Price'] = (data['Close'] - rolling_mean) / rolling_std
        
        # Normalize volume using Series operations
        volume_mean = data['Volume'].rolling(window=20).mean()
        data.loc[:, 'Normalized_Volume'] = data['Volume'] / volume_mean
        
        # Fill NaN values using forward and backward fill
        data = data.bfill().ffill()
        
        return data
    
    # Normalize the entire dataset
    df = normalize_data(df)
    
    # Split data for training and testing (80% train, 20% test)
    train_df = df[:int(len(df) * 0.8)]
    test_df = df[int(len(df) * 0.8):]
    
    print("\nData Statistics:")
    print("Training Data:")
    print(f"Price Range: ${train_df['Close'].min().iloc[0]:.2f} - ${train_df['Close'].max().iloc[0]:.2f}")
    print(f"Average Price: ${train_df['Close'].mean().iloc[0]:.2f}")
    print(f"Price Std Dev: ${train_df['Close'].std().iloc[0]:.2f}")
    print("\nTesting Data:")
    print(f"Price Range: ${test_df['Close'].min().iloc[0]:.2f} - ${test_df['Close'].max().iloc[0]:.2f}")
    print(f"Average Price: ${test_df['Close'].mean().iloc[0]:.2f}")
    print(f"Price Std Dev: ${test_df['Close'].std().iloc[0]:.2f}")
    
    # Create the environment with normalized data
    train_env = TradingEnv(train_df, initial_balance=10000, max_trades=max_trades)
    
    # Initialize PPO agent with modified parameters
    agent = PPOAgent(
        env=train_env,
        model_kwargs={
            "learning_rate": learning_rate,
            "gamma": gamma,
            "n_steps": 2048,  # Increased from 1024 for better learning
            "batch_size": 512,  # Increased from 256 for better learning
            "n_epochs": 20,  # Increased from 10 for better learning
            "gae_lambda": 0.95,  # Increased from 0.92 for better advantage estimation
            "clip_range": 0.4,  # Increased from 0.3 for more exploration
            "ent_coef": 0.1,  # Increased from 0.05 for more exploration
            "vf_coef": 0.5,  # Increased from 0.4 for better value function learning
            "max_grad_norm": 1.0,  # Increased from 0.8 for more aggressive updates
            "use_sde": True,
            "sde_sample_freq": 8,  # Increased from 4 for more exploration
            "target_kl": 0.05,  # Increased from 0.03 for more exploration
            "tensorboard_log": f"{results_dir}/tensorboard/",
            "policy_kwargs": dict(
                net_arch=dict(
                    pi=[256, 256],  # Increased network size
                    vf=[256, 256]   # Increased network size
                ),
                activation_fn=torch.nn.ReLU,
                normalize_images=True,
                log_std_init=1.0  # Increased from 0.8 for more exploration
            )
        }
    )
    
    # Load pre-trained model if specified
    if load_model:
        agent.load(load_model)
        print(f"Loaded pre-trained model from {load_model}")
    
    # Training phase
    if not eval_only and not load_model:
        print("\n--- Training Phase ---")
        agent.train(
            total_timesteps=total_timesteps,
            save_path=results_dir,
            check_freq=5000
        )
        
        # Save final trained model
        agent.save(f"{results_dir}/ppo_final.zip")
        
        # Plot training metrics
        agent.plot_training_metrics(filepath=f"{results_dir}/training_metrics.png")
    
    # Evaluation phase
    print("\n--- Evaluation Phase ---")
    eval_env = TradingEnv(test_df, initial_balance=10000, max_trades=max_trades)
    
    # Run evaluation episode with evaluation mode enabled
    obs, _ = eval_env.reset(options={'evaluation': True})  # Enable evaluation mode
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        reward = reward.item() if hasattr(reward, "item") else float(reward)
        
        total_reward += reward
        steps += 1
        
        if steps % 20 == 0:
            print(f"Eval step {steps}, Portfolio: ${info['portfolio_value']:.2f}")
    
    # Evaluation summary
    print(f"\nEvaluation completed in {steps} steps")
    print(f"Final portfolio value: ${eval_env.portfolio_values[-1]:.2f}")
    print(f"Return: {((eval_env.portfolio_values[-1] / eval_env.initial_balance) - 1) * 100:.2f}%")
    print(f"Max drawdown: {eval_env.max_drawdown * 100:.2f}%")
    print(f"Number of trades: {eval_env.trade_count}")
    
    # Save evaluation results
    eval_env.save_results(f"{results_dir}/evaluation")
    
    # Render evaluation visualization
    eval_env.render()
    plt.savefig(f"{results_dir}/evaluation_portfolio.png")
    
    print(f"\nPPO agent results saved to {results_dir}")
    return agent, eval_env 