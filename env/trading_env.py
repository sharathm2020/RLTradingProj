import gym
import numpy as np
import pandas as pd
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(TradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 1 = holding stock, 0 = no stock

        self.action_space = spaces.Discrete(3)  # 0 = Sell, 1 = Hold, 2 = Buy

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

    def _get_obs(self):
        row = self.df.iloc[self.current_step]

        return np.array([
            float(row['Close'].item() if hasattr(row['Close'], "item") else row['Close']),
            float(row['Volume'].item() if hasattr(row['Volume'], "item") else row['Volume']),
            float(row['SMA_50'].item() if hasattr(row['SMA_50'], "item") else row['SMA_50']),
            float(row['SMA_200'].item() if hasattr(row['SMA_200'], "item") else row['SMA_200']),
            float(self.balance),
            float(self.position),
        ], dtype=np.float32)



    def step(self, action):
        reward = 0
        done = False
        row = self.df.iloc[self.current_step]
        price = float(row['Close'].item() if hasattr(row['Close'], "item") else row['Close'])

        if action == 0 and self.position == 1:
            self.balance += price
            self.position = 0
        elif action == 2 and self.position == 0 and self.balance >= price:
            self.balance -= price
            self.position = 1

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        net_worth = self.balance + (self.position * price)
        reward = net_worth - self.initial_balance

        return self._get_obs(), reward, done, {}


    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        return self._get_obs()
