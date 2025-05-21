from env.trading_env import TradingEnv
from utils.helpers import load_stock_data
from baseline_agent import run_baseline_agent
from agents.q_learning_runner import run_q_learning_agent
from agents.ppo_runner import run_ppo_agent
import argparse

def main():
    parser = argparse.ArgumentParser(description='Trading Environment')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--start', type=str, default='2020-01-01', help='Start date for data')
    parser.add_argument('--end', type=str, default='2023-01-01', help='End date for data')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run (for baseline and Q-learning)')
    parser.add_argument('--timesteps', type=int, default=100000, help='Number of timesteps to train (for PPO)')
    parser.add_argument('--balance', type=float, default=10000, help='Initial balance')
    parser.add_argument('--max_trades', type=int, default=50, help='Maximum trades before penalty')
    parser.add_argument('--mode', type=str, default='baseline', 
                        choices=['baseline', 'qlearning', 'ppo', 'custom'],
                        help='Run mode: baseline, qlearning, ppo, or custom')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load a pre-trained model')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation (no training)')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for future rewards')
    
    args = parser.parse_args()
    
    if args.mode == 'baseline':
        # Run baseline agent
        run_baseline_agent(
            ticker=args.ticker,
            start=args.start,
            end=args.end,
            episodes=args.episodes,
            max_trades=args.max_trades
        )
    elif args.mode == 'qlearning':
        # Run Q-learning agent
        run_q_learning_agent(
            ticker=args.ticker,
            start=args.start,
            end=args.end,
            episodes=args.episodes,
            max_trades=args.max_trades,
            learning_rate=args.lr,
            discount_factor=args.gamma,
            load_model=args.load_model,
            eval_only=args.eval_only
        )
    elif args.mode == 'ppo':
        # Run PPO agent
        run_ppo_agent(
            ticker=args.ticker,
            start=args.start,
            end=args.end,
            total_timesteps=args.timesteps,
            max_trades=args.max_trades,
            learning_rate=args.lr,
            gamma=args.gamma,
            load_model=args.load_model,
            eval_only=args.eval_only
        )
    else:
        # Run custom experiment
        df = load_stock_data(
            ticker=args.ticker,
            start=args.start,
            end=args.end
        )
        
        env = TradingEnv(df, initial_balance=args.balance, max_trades=args.max_trades)
        
        obs = env.reset()
        done = False
        
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            reward = reward.item() if hasattr(reward, "item") else float(reward)
            print(f"Step reward: {reward:.2f}, Portfolio: ${info['portfolio_value']:.2f}")
        
        # Display final results
        print("\nFinal Results:")
        print(f"Portfolio Value: ${env.portfolio_values[-1]:.2f}")
        print(f"Return: {((env.portfolio_values[-1] / env.initial_balance) - 1) * 100:.2f}%")
        print(f"Max Drawdown: {env.max_drawdown * 100:.2f}%")
        print(f"Number of Trades: {env.trade_count}")
        
        # Render visualization
        env.render()
        
        # Save results
        env.save_results()

if __name__ == "__main__":
    main()
