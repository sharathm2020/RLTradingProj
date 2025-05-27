from env.trading_env import TradingEnv
from utils.helpers import load_stock_data
from agents.q_learning import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

def run_q_learning_agent(ticker="AAPL", 
                        start="2020-01-01", 
                        end="2023-01-01", 
                        episodes=100, 
                        max_trades=50,
                        learning_rate=0.1,
                        discount_factor=0.95,
                        exploration_rate=1.0,
                        exploration_decay=0.99,
                        min_exploration_rate=0.1,
                        load_model=None,
                        eval_only=False):
    """
    Run a Q-Learning agent for trading.
    
    Args:
        ticker: Stock ticker symbol
        start: Start date for data
        end: End date for data
        episodes: Number of episodes to run
        max_trades: Maximum number of trades to allow before penalty
        learning_rate: Alpha parameter for learning rate
        discount_factor: Gamma parameter for future reward discount
        exploration_rate: Initial epsilon for exploration
        exploration_decay: Rate at which exploration decreases
        min_exploration_rate: Minimum exploration rate
        load_model: Path to pre-trained model (if None, train new model)
        eval_only: If True, only evaluate (no training)
    """
    print(f"Running Q-Learning agent for {ticker} from {start} to {end}")
    

    # Create results directory
    results_dir = f"results/q_learning_{ticker}_{start}_{end}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    df = load_stock_data(ticker=ticker, start=start, end=end)
    print(f"Loaded {len(df)} data points")
    
    reward_config = {
        "reward_scale": 1.0,
        "drawdown_penalty": 0.5,
        "trade_penalty": 1.0,
        "invalid_action_penalty": 10.0
    }

    # Split data for training and testing (80% train, 20% test)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Create environment for training
    train_env = TradingEnv(train_df, initial_balance=10000, max_trades=max_trades, reward_config=reward_config)
    
    # Initialize Q-Learning agent
    agent = QLearningAgent(
        action_space=train_env.action_space,
        observation_space=train_env.observation_space,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        exploration_rate=exploration_rate,
        exploration_decay=exploration_decay,
        min_exploration_rate=min_exploration_rate
    )
    
    # Load pre-trained model if specified
    if load_model:
        agent.load(load_model)
        print(f"Loaded pre-trained model from {load_model}")
    
    # Training phase
    if not eval_only and not load_model:
        print("\n--- Training Phase ---")
        for episode in range(episodes):
            obs = train_env.reset()
            done = False
            total_reward = 0
            steps = 0
            action_counter = Counter()

            # Run episode
            while not done:
                # Select action using epsilon-greedy policy
                discrete_state = agent.discretize_state(obs)
                q_values = agent.q_table.get(discrete_state, None)
                if q_values is not None:
                    print(f"Step {steps}: Q-values = {q_values}, chosen action = {np.argmax(q_values)}")

                
                action = agent.act(obs)
                action_counter[action] += 1

                # Take action and observe next state and reward
                next_obs, reward, done, info = train_env.step(action)
                reward = reward.item() if hasattr(reward, "item") else float(reward)
                
                # Update Q-values
                agent.learn(obs, action, reward, next_obs, done)
                
                obs = next_obs
                total_reward += reward
                steps += 1
                
                # Print progress periodically
                if steps % 50 == 0:
                    print(f"Episode {episode+1}/{episodes}, Step {steps}, "
                          f"Portfolio: ${info['portfolio_value']:.2f}, "
                          f"Epsilon: {agent.exploration_rate:.4f}")
            
            # Episode summary
            print(f"Episode {episode+1} completed in {steps} steps")
            print(f"Action distribution: (ep {episode+1}): {dict(action_counter)}")
            print(f"Total reward: {total_reward:.2f}")
            print(f"Final portfolio: ${train_env.portfolio_values[-1]:.2f}")
            print(f"Return: {((train_env.portfolio_values[-1] / train_env.initial_balance) - 1) * 100:.2f}%")
            print(f"Exploration rate: {agent.exploration_rate:.4f}")
            
            # Every 10 episodes, save the model
            if (episode + 1) % 10 == 0:
                agent.save(f"{results_dir}/agent_ep{episode+1}.pkl")
        
        # Save final trained model
        agent.save(f"{results_dir}/agent_final.pkl")
        
        # Plot training stats
        agent.plot_training_stats(f"{results_dir}/training_stats.png")
    
    # Evaluation phase
    print("\n--- Evaluation Phase ---")
    eval_env = TradingEnv(test_df, initial_balance=10000, max_trades=max_trades, reward_config=reward_config)
    
    # Run evaluation episode
    obs = eval_env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    # Turn off exploration for evaluation
    orig_exploration = agent.exploration_rate
    agent.exploration_rate = 0
    
    while not done:
        discrete_state = agent.discretize_state(obs)
        q_values = agent.q_table.get(discrete_state, None)
        if q_values is not None:
            print(f"Step {steps}: Q-values = {q_values}, chosen action = {np.argmax(q_values)}")

        action = agent.act(obs)
        obs, reward, done, info = eval_env.step(action)
        reward = reward.item() if hasattr(reward, "item") else float(reward)
        
        total_reward += reward
        steps += 1
        
        if steps % 20 == 0:
            print(f"Eval step {steps}, Portfolio: ${info['portfolio_value']:.2f}")
    
    # Restore exploration rate
    agent.exploration_rate = orig_exploration
    
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
    
    print(f"\nQ-Learning agent results saved to {results_dir}")

    plt.figure(figsize=(10, 5))
    plt.plot(eval_env.portfolio_values, label="Portfolio Value")
    plt.title("Evaluation Portfolio Value Over Time")
    plt.xlabel("Step")
    plt.ylabel("Portfolio ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{results_dir}/evaluation_equity_curve.png")
    print(f"Saved evaluation equity curve to evaluation_equity_curve.png")


    return agent, eval_env 