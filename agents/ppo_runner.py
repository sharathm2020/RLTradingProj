from env.trading_env import TradingEnv
from utils.helpers import load_stock_data
from agents.ppo_agent import PPOAgent, CustomCallback
import numpy as np
import matplotlib.pyplot as plt
import os

def run_ppo_agent(ticker="AAPL", 
                 start="2020-01-01", 
                 end="2023-01-01", 
                 total_timesteps=100000,
                 max_trades=50,
                 learning_rate=0.0003,
                 gamma=0.99,
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
    
    # Split data for training and testing (80% train, 20% test)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Create environment for training
    train_env = TradingEnv(train_df, initial_balance=10000, max_trades=max_trades)
    
    # Initialize PPO agent
    agent = PPOAgent(
        env=train_env,
        model_kwargs={
            "learning_rate": learning_rate,
            "gamma": gamma,
            "verbose": 1
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
    
    # Run evaluation episode
    obs = eval_env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
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