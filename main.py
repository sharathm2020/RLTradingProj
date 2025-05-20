from env.trading_env import TradingEnv
from utils.helpers import load_stock_data

df = load_stock_data()
env = TradingEnv(df)

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    reward = reward.item() if hasattr(reward, "item") else float(reward)
    print(f"Step reward: {reward:.2f}")
