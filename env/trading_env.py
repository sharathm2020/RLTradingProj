import gymnasium as gym
import numpy as np
import pandas as pd
from gym import spaces
import matplotlib.pyplot as plt
from collections import deque
import os
import json

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, max_trades=None, transaction_cost=0.001, reward_config=None):
        super(TradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 1 = holding stock, 0 = no stock
        self.max_trades = max_trades  # Maximum number of trades allowed
        self.transaction_cost = transaction_cost  # Transaction cost as percentage
        
        # History tracking
        self.trades = []
        self.portfolio_values = []
        self.max_portfolio_value = initial_balance
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.trade_count = 0

        self.action_space = spaces.Discrete(3)  # 0 = Sell, 1 = Hold, 2 = Buy

        # Add drawdown to observation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        self.reward_config = reward_config or {
            "reward_scale": 1.0,
            "drawdown_penalty": 0.5,
            "trade_penalty": 1.0,
            "invalid_action_penalty": 10.0
        }

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        price = float(row['Close'].item() if hasattr(row['Close'], "item") else row['Close'])
        
        # Include current drawdown in the observation
        return np.array([
            float(row['Close'].item() if hasattr(row['Close'], "item") else row['Close']),
            float(row['Volume'].item() if hasattr(row['Volume'], "item") else row['Volume']),
            float(row['SMA_50'].item() if hasattr(row['SMA_50'], "item") else row['SMA_50']),
            float(row['SMA_200'].item() if hasattr(row['SMA_200'], "item") else row['SMA_200']),
            float(self.balance),
            float(self.position),
            float(self.current_drawdown),
        ], dtype=np.float32)

    def step(self, action):
        reward = 0
        done = False
        row = self.df.iloc[self.current_step]
        price = float(row['Close'].item() if hasattr(row['Close'], "item") else row['Close'])
        date = row.name if hasattr(row, 'name') else self.current_step

        # Initialize info dictionary to track step details
        info = {
            'date': date,
            'price': price,
            'action': action,
            'trade_made': False,
            'trade_type': None
        }

        # Apply transaction costs and update positions based on action
        if action == 0 and self.position == 1:  # Sell
            # Apply transaction cost
            transaction_fee = price * self.transaction_cost
            self.balance += (price - transaction_fee)
            self.position = 0
            self.trade_count += 1
            
            # Log the trade
            trade = {
                'date': date,
                'type': 'SELL',
                'price': price,
                'fee': transaction_fee,
                'balance_after': self.balance
            }
            self.trades.append(trade)
            
            info['trade_made'] = True
            info['trade_type'] = 'SELL'
            
            # Print trade details
            print(f"SELL at ${price:.2f}, Fee: ${transaction_fee:.2f}, Balance: ${self.balance:.2f}")
            
        elif action == 2 and self.position == 0 and self.balance >= price:  # Buy
            # Apply transaction cost
            transaction_fee = price * self.transaction_cost
            self.balance -= (price + transaction_fee)
            self.position = 1
            self.trade_count += 1
            
            # Log the trade
            trade = {
                'date': date,
                'type': 'BUY',
                'price': price,
                'fee': transaction_fee,
                'balance_after': self.balance
            }
            self.trades.append(trade)
            
            info['trade_made'] = True
            info['trade_type'] = 'BUY'
            
            # Print trade details
            print(f"BUY at ${price:.2f}, Fee: ${transaction_fee:.2f}, Balance: ${self.balance:.2f}")

        # Calculate portfolio value
        portfolio_value = self.balance + (self.position * price)
        self.portfolio_values.append(portfolio_value)
        
        # Update max portfolio value and calculate drawdown
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
        
        # Calculate current drawdown
        if self.max_portfolio_value > 0:
            self.current_drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        info['portfolio_value'] = portfolio_value
        info['drawdown'] = self.current_drawdown
        info['max_drawdown'] = self.max_drawdown
        
        # Move to next step
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
        
        # Calculate base reward: change in portfolio value
        #base_reward = (portfolio_value - self.initial_balance) * self.reward_config.get("reward_scale", 1.0)
        
        base_reward = (portfolio_value - self.initial_balance) * 2.0

        # Drawdown penalty (scaled)
        drawdown_penalty = self.current_drawdown * self.reward_config.get("drawdown_penalty", 0.0)

        # Excessive trading penalty
        trading_penalty = 0
        if self.max_trades and self.trade_count > self.max_trades:
            trades_over_limit = self.trade_count - self.max_trades
            trading_penalty = trades_over_limit * self.reward_config.get("trade_penalty", 0.0)

        # Optional profit signal for SELLs
        trade_profit = 0
        if info['trade_type'] == 'SELL':
            last_trade = self.trades[-2] if len(self.trades) >= 2 else None
            if last_trade and last_trade['type'] == 'BUY':
                trade_profit = price - last_trade['price']


        # Penalize invalid actions
        if (action == 2 and self.position == 1) or (action == 0 and self.position == 0):
            reward -= self.reward_config.get("invalid_action_penalty", 10)
            info["invalid_action_penalty_applied"] = True
        else:
            info["invalid_action_penalty_applied"] = False

        # Final reward
        reward = (
            trade_profit * 10
            + base_reward
            - drawdown_penalty
            - trading_penalty
        )
        
        info['base_reward'] = base_reward
        info['drawdown_penalty'] = drawdown_penalty
        info['trading_penalty'] = trading_penalty
        info['final_reward'] = reward
        
        return self._get_obs(), reward, done, info

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        self.portfolio_values = []
        self.max_portfolio_value = self.initial_balance
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.trade_count = 0
        return self._get_obs()
        
    def render(self, mode='human'):
        """
        Render the environment.
        If mode is 'human', display plots of portfolio value and drawdown.
        """
        if len(self.portfolio_values) < 2:
            print("Not enough data to render")
            return
        
        # For plotting dates on x-axis
        if hasattr(self.df.iloc[0], 'name') and isinstance(self.df.iloc[0].name, pd.Timestamp):
            dates = [self.df.iloc[i].name for i in range(len(self.portfolio_values))]
        else:
            dates = list(range(len(self.portfolio_values)))
        
        plt.figure(figsize=(12, 8))
        
        # Portfolio value subplot
        plt.subplot(2, 1, 1)
        plt.plot(dates, self.portfolio_values, label='Portfolio Value', color='blue')
        
        # Mark buy and sell points
        for trade in self.trades:
            if trade['type'] == 'BUY':
                idx = dates.index(trade['date']) if trade['date'] in dates else None
                if idx is not None:
                    plt.scatter(dates[idx], self.portfolio_values[idx], color='green', marker='^', s=100)
            elif trade['type'] == 'SELL':
                idx = dates.index(trade['date']) if trade['date'] in dates else None
                if idx is not None:
                    plt.scatter(dates[idx], self.portfolio_values[idx], color='red', marker='v', s=100)
        
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date/Steps')
        plt.ylabel('Value ($)')
        plt.grid(True)
        plt.legend()
        
        # Drawdown subplot
        plt.subplot(2, 1, 2)
        drawdowns = []
        for i in range(len(self.portfolio_values)):
            max_value_so_far = max(self.portfolio_values[:i+1])
            current_value = self.portfolio_values[i]
            drawdown = (max_value_so_far - current_value) / max_value_so_far if max_value_so_far > 0 else 0
            drawdowns.append(drawdown)
        
        plt.plot(dates, drawdowns, label='Drawdown', color='red')
        plt.title('Drawdown Over Time')
        plt.xlabel('Date/Steps')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def save_results(self, filepath='results'):
        """
        Save performance results to disk.
        """
        # Create directory if it doesn't exist
        os.makedirs(filepath, exist_ok=True)
        
        # Prepare results data
        results = {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'final_portfolio_value': self.portfolio_values[-1] if self.portfolio_values else self.balance,
            'return_percentage': ((self.portfolio_values[-1] / self.initial_balance) - 1) * 100 if self.portfolio_values else 0,
            'max_drawdown': self.max_drawdown * 100,  # Convert to percentage
            'trade_count': self.trade_count,
            'trades': self.trades,
        }
        
        # Save as JSON
        with open(os.path.join(filepath, 'trading_results.json'), 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        # Save portfolio values as CSV
        if hasattr(self.df.iloc[0], 'name') and isinstance(self.df.iloc[0].name, pd.Timestamp):
            dates = [self.df.iloc[i].name for i in range(min(len(self.portfolio_values), len(self.df)))]
            df_results = pd.DataFrame({
                'date': dates[:len(self.portfolio_values)],
                'portfolio_value': self.portfolio_values
            })
        else:
            df_results = pd.DataFrame({
                'step': list(range(len(self.portfolio_values))),
                'portfolio_value': self.portfolio_values
            })
        
        df_results.to_csv(os.path.join(filepath, 'portfolio_values.csv'), index=False)
        
        # Create and save plots
        self.render()
        plt.savefig(os.path.join(filepath, 'performance_plot.png'))
        
        print(f"Results saved to {filepath}")
